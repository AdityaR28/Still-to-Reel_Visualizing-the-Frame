import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous CUDA calls for better error reporting

import argparse
import os.path
import json
import imageio
import torch
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F

from lib.config import load_config
from lib.utils.general_utils import *
from lib.model.inpaint.model import SpaceTimeAnimationModel
from lib.utils.render_utils import *
from lib.model.motion.motion_model import SPADEUnetMaskMotion
from lib.model.motion.sync_batchnorm import convert_model
from lib.renderer import ImgRenderer
from lib.model.inpaint.inpainter import Inpainter
from lib.utils.data_utils import resize_img
from third_party.DPT.run_monodepth import run_dpt


def runge_kutta_integration(flow, t_step, dt=0.02):
    """
    Apply 4th-order Runge-Kutta integration to the flow field for better accuracy.
    :param flow: The current flow field (tensor).
    :param t_step: The current time step.
    :param dt: The time step size for the integration (default 0.02).
    :return: The updated flow field after integration.
    """
    k1 = dt * flow
    k2 = dt * (flow + 0.5 * k1)
    k3 = dt * (flow + 0.5 * k2)
    k4 = dt * (flow + k3)
    updated_flow = flow + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return updated_flow


def clamp_flow(flow, max_magnitude=5.0):
    """
    Clamp the flow vector magnitude so it doesn't exceed a given threshold.
    :param flow: Flow field tensor of shape (1, 2, H, W).
    :param max_magnitude: Maximum allowed magnitude of the flow vectors.
    :return: Flow field with clamped magnitudes.
    """
    # Compute the norm per-pixel; shape [1,1,H,W]
    flow_norm = (flow**2).sum(dim=1, keepdim=True).sqrt()
    # Create a scaling factor of the same shape
    scale = torch.ones_like(flow_norm)
    mask = flow_norm > max_magnitude
    scale[mask] = max_magnitude / flow_norm[mask]
    # Multiply flow by scale (broadcasted along the channel dimension)
    return flow * scale


def bilateral_filter(depth_map, diameter=7, sigma_color=100, sigma_space=100):
    """
    Apply bilateral filtering to a depth map to preserve edges and smooth out noise.
    :param depth_map: Input depth map.
    :param diameter: Diameter of the pixel neighborhood.
    :param sigma_color: Filter sigma in color space.
    :param sigma_space: Filter sigma in coordinate space.
    :return: Bilateral filtered depth map.
    """
    import cv2
    depth_map = np.float32(depth_map)
    return cv2.bilateralFilter(depth_map, diameter, sigma_color, sigma_space)


def smooth_flow(flow, kernel_size=5, sigma=2.0):
    """
    Apply a Gaussian smoothing to the flow field to reduce artifacts.
    :param flow: Input flow field of shape (1, 2, H, W).
    :param kernel_size: Size of the Gaussian kernel.
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Smoothed flow field.
    """
    device = flow.device
    grid = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    gaussian_1d = torch.exp(-grid**2 / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    kernel = torch.outer(gaussian_1d, gaussian_1d).view(1, 1, kernel_size, kernel_size)
    
    flow0 = F.conv2d(flow[:, 0:1], kernel, padding=kernel_size // 2)
    flow1 = F.conv2d(flow[:, 1:2], kernel, padding=kernel_size // 2)
    return torch.cat([flow0, flow1], dim=1)


def generate_mask_hints_from_user(args, config):
    json_file = os.path.join(args.input_dir, 'image.json')
    mask_file = os.path.join(args.input_dir, 'image_json', 'mask.png')

    mask = imageio.imread(mask_file)
    height, width = mask.shape[0], mask.shape[1]

    hint_y = []
    hint_x = []
    hint_motion = []

    data = json.load(open(json_file))
    for shape in data['shapes']:
        if shape['label'].startswith('hint'):
            points = np.array(shape["points"])
            if len(points) < 2:
                print(f"Skipping shape with insufficient points: {shape['points']}")
                continue
            start, end = points[:2]
            hint_x.append(int(start[0]))
            hint_y.append(int(start[1]))
            hint_motion.append((end - start) / 50.)

    hint_y = torch.tensor(hint_y)
    hint_x = torch.tensor(hint_x)
    hint_motion = torch.tensor(np.array(hint_motion)).permute(1, 0)[None]
    max_hint = hint_motion.shape[-1]
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    xs = xs.view(1, 1, width).repeat(1, height, 1)
    ys = ys.view(1, height, 1).repeat(1, 1, width)
    xys = torch.cat((xs, ys), 1).view(2, -1)

    dense_motion = torch.zeros(1, 2, height * width)
    dense_motion_norm = torch.zeros(dense_motion.shape).view(1, 2, -1)

    sigma = max(10, height // (max_hint + 1))
    hint_y = hint_y.long()
    hint_x = hint_x.long()
    for i_hint in range(max_hint):
        dist = ((xys - xys.view(2, height, width)[:, hint_y[i_hint], hint_x[i_hint]].unsqueeze(1)) ** 2).sum(0, True).sqrt()
        weight = (-(dist / sigma) ** 2).exp().unsqueeze(0)
        dense_motion += weight * hint_motion[:, :, i_hint].unsqueeze(2)
        dense_motion_norm += weight
    dense_motion_norm[dense_motion_norm == 0.0] = 1.0
    dense_motion = dense_motion / dense_motion_norm
    dense_motion = dense_motion.view(1, 2, height, width) * torch.tensor(mask).bool()

    hint = dense_motion
    hint_scale = [config['W'] / width, config['W'] / height]
    hint = hint * torch.FloatTensor(hint_scale).view(1, 2, 1, 1)
    hint = F.interpolate(hint, (config['W'], config['W']), mode='bilinear', align_corners=False)
    mask = F.interpolate(torch.tensor(mask[None, None]).bool().float(), (config['W'], config['W']), mode='area')

    return mask, hint


def get_input_data(args, config, video_out_folder, ds_factor=1):
    motion_input_transform = Compose([
        torchvision.transforms.Resize((config['motionH'], config['motionW']),
                                        InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    to_tensor = ToTensor()

    try:
        img_file = os.path.join(args.input_dir, 'image.png')
        motion_rgb = Image.open(img_file)
    except:
        img_file = os.path.join(args.input_dir, 'image.jpg')
        motion_rgb = Image.open(img_file)
    motion_rgb = motion_input_transform(motion_rgb)
    mask, hints = generate_mask_hints_from_user(args, config)

    dpt_out_dir = os.path.join(video_out_folder, 'dpt_depth')

    src_img = imageio.imread(img_file) / 255.
    src_img = resize_img(src_img, ds_factor)

    h, w = src_img.shape[:2]

    dpt_model_path = 'ckpts/dpt_hybrid-midas-501f0c75.pt'
    run_dpt(input_path=args.input_dir, output_path=dpt_out_dir, model_path=dpt_model_path, optimize=False)
    disp_file = os.path.join(dpt_out_dir, 'image.png')

    src_disp = imageio.imread(disp_file) / 65535.
    src_disp = bilateral_filter(src_disp)

    src_depth = 1. / np.maximum(src_disp, 1e-6)

    import cv2
    src_depth = cv2.GaussianBlur(src_depth, (3, 3), 0)

    src_depth = resize_img(src_depth, ds_factor)
    if not np.all(np.isfinite(src_depth)):
        finite_vals = src_depth[np.isfinite(src_depth)]
        if finite_vals.size > 0:
            src_depth = np.where(np.isfinite(src_depth), src_depth, finite_vals.max())
        else:
            src_depth = np.zeros_like(src_depth)

    intrinsic = np.array([
        [max(h, w), 0, w // 2],
        [0, max(h, w), h // 2],
        [0, 0, 1]
    ])

    pose = np.eye(4)

    return {
        'motion_rgbs': motion_rgb[None, ...],
        'src_img': to_tensor(src_img).float()[None],
        'src_depth': to_tensor(src_depth).float()[None],
        'hints': hints[0],
        'mask': mask[0],
        'intrinsic': torch.from_numpy(intrinsic).float()[None],
        'pose': torch.from_numpy(pose).float()[None],
        'scale_shift': torch.tensor([1., 0.]).float()[None],
        'src_rgb_file': [img_file],
    }


def render(args):
    device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)

    video_out_folder = os.path.join(args.input_dir, 'output')
    os.makedirs(video_out_folder, exist_ok=True)

    check_file(args.config)
    config = load_config(args.config)

    data = get_input_data(args, config['data'], video_out_folder, ds_factor=args.ds_factor)
    torch.cuda.empty_cache()

    ###########################################################################
    """ model """
    model = SpaceTimeAnimationModel(args, config)
    if model.start_step == 0:
        raise Exception('No pretrained model found! Please check the model path.')

    scene_flow_estimator = SPADEUnetMaskMotion(config['generator']).to(device)
    scene_flow_estimator = convert_model(scene_flow_estimator)
    scene_flow_estimator_weight = torch.load('ckpts/sceneflow_model.pth',
                                             map_location=torch.device(device))
    scene_flow_estimator.load_state_dict(scene_flow_estimator_weight['netG'])
    inpainter = Inpainter(device=device)
    renderer = ImgRenderer(args, config, model, scene_flow_estimator, inpainter, device)

    ###########################################################################
    """ render """
    model.switch_to_eval()
    with torch.no_grad():
        renderer.process_data(data)

        coord, flow, pts_src, featmaps_src, rgba_layers_src, depth_layers_src, mask_layers_src = \
            renderer.compute_flow_and_inpaint()

        fs = args.flow_scale if args.flow_scale != 1.0 else 0.8
        flow = flow / fs

        flow = runge_kutta_integration(flow, t_step=0)
        flow = clamp_flow(flow, max_magnitude=5.0)
        for _ in range(3):
            flow = smooth_flow(flow, kernel_size=5, sigma=2.0)
        flow = flow.contiguous()

        num_frames = [45, 45, 45, 60]
        video_paths = ['up-down', 'zoom-in', 'side', 'circle']
        Ts = [
            define_camera_path(num_frames[0], 0., -0.08, 0., path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[1], 0., 0., -0.24, path_type='straight-line', return_t_only=True),
            define_camera_path(num_frames[2], -0.09, 0, -0, path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[3], -0.04, -0.04, -0.09, path_type='circle', return_t_only=True),
        ]

        # Removed hard cropping to preserve full resolution.
        kernel = torch.ones(5, 5, device=device)

        frames = []
        for j, T in enumerate(Ts):
            T = torch.from_numpy(T).float().to(renderer.device)
            time_steps = range(0, num_frames[j])
            start_index = torch.tensor([0]).to(device)
            end_index = torch.tensor([num_frames[j] - 1]).to(device)
            for middle_index, t_step in tqdm(enumerate(time_steps), total=len(time_steps), ncols=150,
                                             desc='generating video of {} camera trajectory'.format(video_paths[j])):
                middle_index = torch.tensor([middle_index]).to(device)
                time = ((middle_index.float() - start_index.float()).float() /
                        (end_index.float() - start_index.float() + 1.0).float()).item()

                flow_f = renderer.euler_integration(flow, middle_index.long() - start_index.long())
                flow_b = renderer.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
                flow_f = flow_f.permute(0, 2, 3, 1)
                flow_b = flow_b.permute(0, 2, 3, 1)

                _, all_pts_f, _, all_rgbas_f, _, all_feats_f, \
                    all_masks_f, all_optical_flow_f = \
                    renderer.compute_scene_flow_for_motion(coord, torch.inverse(renderer.pose), renderer.src_img,
                                                           rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                           mask_layers_src, flow_f, kernel, with_inpainted=True)

                _, all_pts_b, _, all_rgbas_b, _, all_feats_b, \
                    all_masks_b, all_optical_flow_b = \
                    renderer.compute_scene_flow_for_motion(coord, torch.inverse(renderer.pose), renderer.src_img,
                                                           rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                           mask_layers_src, flow_b, kernel, with_inpainted=True)

                all_pts_flowed = torch.cat(all_pts_f + all_pts_b)
                all_rgbas_flowed = torch.cat(all_rgbas_f + all_rgbas_b)
                all_feats_flowed = torch.cat(all_feats_f + all_feats_b)
                all_masks = torch.cat(all_masks_f + all_masks_b)
                all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)
                num_pts_2 = sum([len(x) for x in all_pts_b])
                all_side_ids[-num_pts_2:] = 1

                pred_img, _, meta = renderer.render_pcd(all_pts_flowed,
                                                        all_rgbas_flowed,
                                                        all_feats_flowed,
                                                        all_masks, all_side_ids,
                                                        t=T[middle_index.item()],
                                                        time=time,
                                                        t_step=t_step,
                                                        path_type=video_paths[j])
                torch.cuda.synchronize()

                frame = (255. * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                # Full resolution output; final filtering is disabled to preserve clarity.
                frames.append(frame)

            video_out_file = os.path.join(video_out_folder,
                                          f'{video_paths[j]}_flow_scale={args.flow_scale}.mp4')
            try:
                imageio.mimwrite(video_out_file, frames, fps=25, quality=10)
                print(f"Video saved successfully: {video_out_file}")
            except Exception as e:
                print(f"Error saving video {video_out_file}: {e}")

        print(f'Space-time videos have been saved in {video_out_folder}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ########## general ##########
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('--input_dir', type=str, help='input folder that contains src images', required=True)
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')

    parser.add_argument('--save_frames', action='store_true', help='if save frames')
    parser.add_argument('--correct_inpaint_depth', action='store_true',
                        help='use this option to correct the depth of inpainting area to prevent occlusion')
    parser.add_argument("--flow_scale", type=float, default=1.0,
                        help='flow scale that used to control the speed of fluid')
    parser.add_argument("--ds_factor", type=float, default=1.0,
                        help='downsample factor for the input images')

    ########## checkpoints ##########
    parser.add_argument("--ckpt_path", type=str, default='',
                        help='specific weights file to reload')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')
    args = parser.parse_args()

    render(args)
