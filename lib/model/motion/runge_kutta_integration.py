import torch
import torch.nn as nn

def runge_kutta_integration(motion, destination_frame, dt=1, return_all_frames=False):
    """
    Runge-Kutta integration (4th order) to compute displacement maps over time for each time step.
    
    :param motion: Motion field (Bx2xHxW).
    :param destination_frame: Number of time steps to process.
    :param dt: Time step to scale the motion values (acts as a time step like in Euler).
    :param return_all_frames: If True, return displacement maps for all intermediate frames.
    :return: Displacement map and optionally all frames or final coordinates.
    """
    assert motion.dim() == 4, "Motion must be 4D (Bx2xHxW)"
    b, c, height, width = motion.shape
    assert b == 1, "Function implemented for batch size = 1"
    assert c == 2, f"Input motion field should be Bx2xHxW. Given tensor: {motion.shape}"

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=motion.device),
        torch.linspace(0, width - 1, width, device=motion.device),
        indexing="ij"
    )
    coord = torch.stack([x, y], dim=0).long()

    destination_coords = coord.clone().float()
    destination_coords_ = destination_coords.clone().float()

    if return_all_frames:
        displacements = torch.zeros(destination_frame + 1, 2, height, width, device=motion.device)
        visible_pixels = torch.ones(destination_frame + 1, 1, height, width, device=motion.device)
    else:
        displacements = torch.zeros(1, 2, height, width, device=motion.device)
        visible_pixels = torch.ones(1, 1, height, width, device=motion.device)
    invalid_mask = torch.zeros(1, height, width, device=motion.device).bool()

    for frame_id in range(1, destination_frame + 1):
        # Runge-Kutta calculations for k1, k2, k3, k4
        k1 = dt * motion[0][:, destination_coords[1].round().long(), destination_coords[0].round().long()]
        k2 = dt * motion[0][:, 
            (destination_coords[1] + 0.5 * k1[1]).round().long(), 
            (destination_coords[0] + 0.5 * k1[0]).round().long()]
        k3 = dt * motion[0][:, 
            (destination_coords[1] + 0.5 * k2[1]).round().long(), 
            (destination_coords[0] + 0.5 * k2[0]).round().long()]
        k4 = dt * motion[0][:, 
            (destination_coords[1] + k3[1]).round().long(), 
            (destination_coords[0] + k3[0]).round().long()]

        # Update displacement using RK formula
        displacement = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        destination_coords += displacement

        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()

        # DEBUG
        destination_coords_ = destination_coords_ + displacement

        if return_all_frames:
            displacements[frame_id] = (destination_coords_ - coord.float()).unsqueeze(0)
            displacements[frame_id][invalid_mask] = torch.max(height, width) + 1
            visible_pixels[frame_id] = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
        else:
            displacements = (destination_coords_ - coord.float()).unsqueeze(0)
            visible_pixels = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)

    return displacements, visible_pixels


class RungeKuttaIntegration(nn.Module):
    def __init__(self, dt=1):
        super().__init__()
        self.dt = dt

    def forward(self, motion, destination_frame, return_all_frames=False, show_visible_pixels=False, trajectory_type =None):
        displacements = torch.zeros(motion.shape).to(motion.device)
        visible_pixels = torch.zeros(motion.shape[0], 1, motion.shape[2], motion.shape[3], device=motion.device)

        for b in range(motion.shape[0]):
            displacements[b:b + 1], visible_pixels[b:b + 1] = runge_kutta_integration(
                motion[b:b + 1], destination_frame[b], dt=self.dt, return_all_frames=return_all_frames)

        if show_visible_pixels:
            return displacements, visible_pixels
        else:
            return displacements
