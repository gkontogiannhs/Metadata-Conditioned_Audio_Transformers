import random
import torch
import numpy as np
from typing import Tuple


class SpecAugment(torch.nn.Module):
    """
    SpecAugment with probability p for applying augmentation.
    """

    def __init__(self, policy: str = 'LB', mask: str = 'mean', p: float = None) -> None:
        super().__init__()

        # Set policy-specific parameters
        if policy == 'LB':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif policy == 'LD':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif policy == 'SM':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif policy == 'SS':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 40, 27, 2, 70, 0.2, 2
        elif policy == 'icbhi_sup':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 0, 20, 2, 50, 1.0, 2
        elif policy == 'icbhi_ast_sup':
            self.W, self.F, self.m_F, self.T, policy_p, self.m_T = 0, 48, 2, 160, 1.0, 2
        else:
            raise ValueError(f"Unsupported policy '{policy}'.")

        # allow override
        self.p = p if p is not None else policy_p
        self.policy = policy
        self.mask = mask

    def time_warp(self) -> torch.Tensor:
        """
        Apply time warping to the mel spectrogram using sparse image warp.

        Returns:
            torch.Tensor: Time-warped mel spectrogram of shape [channel, freq, time].

        Notes:
            - The input mel spectrogram is expected to be stored in self.mel_spectrogram
              with shape [channel, freq, time].
            - Uses control points to warp the spectrogram along the time axis.
        """
        num_rows = self.mel_spectrogram.shape[2]  # time dimension
        spec_len = self.mel_spectrogram.shape[1]  # freq dimension
        device = self.mel_spectrogram.device

        # Random point along time axis within bounds [W, num_rows - W]
        pt = (num_rows - 2 * self.W) * torch.rand([1], dtype=torch.float, device=device) + self.W

        # Control points on freq axis (0 to spec_len//2)
        src_ctr_pt_freq = torch.arange(0, spec_len // 2, device=device)
        # Time control points all at pt
        src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq, dtype=torch.float, device=device) * pt
        src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1).float()

        # Destination control points shifted by w in time axis
        w = 2 * self.W * torch.rand([1], dtype=torch.float, device=device) - self.W
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1).float()

        source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  # (1, freq//2, 2)
        dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  # (1, freq//2, 2)

        warped_spectro, _ = sparse_image_warp(self.mel_spectrogram, source_control_point_locations, dest_control_point_locations)

        return warped_spectro.squeeze(3)

    def freq_mask(self) -> torch.Tensor:
        """
        Apply frequency masking to the mel spectrogram.

        Returns:
            torch.Tensor: Frequency masked mel spectrogram of shape [channel, freq, time].

        Notes:
            - Masks m_F frequency bands randomly.
            - Masking value is either mean or zero depending on self.mask.
        """
        if self.mask == 'mean':
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            mask_value = 0.0
        else:
            raise ValueError(f"Unsupported mask value '{self.mask}'. Use 'mean' or 'zero'.")

        v = self.mel_spectrogram.shape[1]  # number of mel bins (freq)

        for _ in range(self.m_F):
            f = int(np.random.uniform(0, self.F))  # mask width
            f0 = random.randint(0, v - f)          # start freq bin
            self.mel_spectrogram[:, f0:f0 + f, :] = mask_value

        return self.mel_spectrogram

    def time_mask(self) -> torch.Tensor:
        """
        Apply time masking to the mel spectrogram.

        Returns:
            torch.Tensor: Time masked mel spectrogram of shape [channel, freq, time].

        Notes:
            - Masks m_T time intervals randomly.
            - Masking value is either mean or zero depending on self.mask.
        """
        if self.mask == 'mean':
            mask_value = self.mel_spectrogram.mean()
        elif self.mask == 'zero':
            mask_value = 0.0
        else:
            raise ValueError(f"Unsupported mask value '{self.mask}'. Use 'mean' or 'zero'.")

        tau = self.mel_spectrogram.shape[2]  # time frames

        for _ in range(self.m_T):
            t = int(np.random.uniform(0, self.T))  # mask width
            t0 = random.randint(0, tau - t)        # start time frame
            self.mel_spectrogram[:, :, t0:t0 + t] = mask_value

        return self.mel_spectrogram


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape [freq, time] or [channel, freq, time]
        Returns:
            Augmented tensor of shape [channel, freq, time]
        """
        # Ensure [channel, freq, time]
        if img.dim() == 2:           # [freq, time] -> grayscale/spectrogram-like inputs
            img = img.unsqueeze(0)   # [1, freq, time]
        self.mel_spectrogram = img.clone()

        if self.p >= torch.rand(1).item():
            if self.W > 0:
                try:
                    self.mel_spectrogram = self.time_warp()
                except Exception:
                    pass
            self.mel_spectrogram = self.freq_mask()
            self.mel_spectrogram = self.time_mask()

        return self.mel_spectrogram # returns as [channel, freq,  time]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy='{self.policy}', mask='{self.mask}')"


def sparse_image_warp(
    img_tensor: torch.Tensor,
    source_control_points: torch.Tensor,
    dest_control_points: torch.Tensor,
    image_height: int,
    image_width: int,
    interpolation_order: int = 2,
    regularization_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp an image tensor using sparse control points and polyharmonic spline interpolation.

    Args:
        img_tensor: Tensor of shape [batch_size, image_height, image_width].
        source_control_points: Tensor of shape [batch_size, num_control_points, 2].
        dest_control_points: Tensor of shape [batch_size, num_control_points, 2].
        image_height: Height of the image.
        image_width: Width of the image.
        interpolation_order: Order of the polyharmonic spline interpolation.
        regularization_weight: Regularization weight for spline fitting.

    Returns:
        warped_image: Tensor of warped image of shape [batch_size, image_height, image_width].
        dense_flows: Tensor of dense flow vectors of shape [batch_size, image_height, image_width, 2].
    """
    device = img_tensor.device
    control_point_flows = dest_control_points - source_control_points
    batch_size = img_tensor.shape[0]

    flattened_grid_locations = get_flat_grid_locations(image_height, image_width, device)

    flattened_flows = interpolate_spline(
        dest_control_points,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight,
    )

    dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)

    warped_image = dense_image_warp(img_tensor, dense_flows)

    return warped_image, dense_flows


def get_grid_locations(
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a grid of (y, x) pixel coordinates.

    Args:
        image_height: Height of the image.
        image_width: Width of the image.
        device: Torch device.

    Returns:
        Tensor of shape [image_height, image_width, 2] containing (y, x) coordinates.
    """
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((y_grid, x_grid), dim=-1)


def flatten_grid_locations(
    grid_locations: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten a grid of locations to a 2D tensor.

    Args:
        grid_locations: Tensor of shape [height, width, 2].

    Returns:
        Tensor of shape [height * width, 2].
    """
    return torch.reshape(grid_locations, [-1, 2])


def get_flat_grid_locations(
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a flattened grid of (y, x) pixel coordinates.

    Args:
        image_height: Height of the image.
        image_width: Width of the image.
        device: Torch device.

    Returns:
        Tensor of shape [image_height * image_width, 2].
    """
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    return torch.stack((y_grid, x_grid), dim=-1).reshape(image_height * image_width, 2)


def create_dense_flows(
    flattened_flows: torch.Tensor,
    batch_size: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """
    Reshape flattened flow vectors into a dense flow field.

    Args:
        flattened_flows: Tensor of shape [batch_size, image_height * image_width, 2].
        batch_size: Batch size.
        image_height: Height of the image.
        image_width: Width of the image.

    Returns:
        Tensor of shape [batch_size, image_height, image_width, 2].
    """
    return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])


def interpolate_spline(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    query_points: torch.Tensor,
    order: int,
    regularization_weight: float = 0.0,
) -> torch.Tensor:
    """
    Interpolate values at query points using polyharmonic spline interpolation.

    Args:
        train_points: Tensor of shape [batch_size, n, d], control points.
        train_values: Tensor of shape [batch_size, n, k], values at control points.
        query_points: Tensor of shape [m, d], points to query.
        order: Order of the polyharmonic spline.
        regularization_weight: Regularization weight.

    Returns:
        Tensor of shape [batch_size, m, k], interpolated values at query points.
    """
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    query_values = apply_interpolation(query_points, train_points, w, v, order)
    return query_values


def solve_interpolation(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    order: int,
    regularization_weight: float,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the polyharmonic spline interpolation system.

    Args:
        train_points: Tensor of shape [batch_size, n, d].
        train_values: Tensor of shape [batch_size, n, k].
        order: Interpolation order.
        regularization_weight: Regularization weight.
        eps: Small value to stabilize matrix.

    Returns:
        Tuple of tensors (w, v) representing spline coefficients.
    """
    device = train_points.device
    batch_size, n, d = train_points.shape
    k = train_values.shape[-1]

    c = train_points
    f = train_values.float()

    matrix_a = phi(cross_squared_distance_matrix(c, c), order).unsqueeze(0)  # [b, n, n]

    # Regularization can be added here if needed (currently disabled).

    ones = torch.ones(batch_size, n, 1, dtype=train_points.dtype, device=device)
    matrix_b = torch.cat((c, ones), dim=2).float()  # [b, n, d + 1]

    left_block = torch.cat((matrix_a, matrix_b.transpose(1, 2)), dim=1)  # [b, n + d + 1, n]

    num_b_cols = matrix_b.shape[2]  # d + 1

    lhs_zeros = torch.randn(batch_size, num_b_cols, num_b_cols, device=device) * eps
    right_block = torch.cat((matrix_b, lhs_zeros), dim=1)  # [b, n + d + 1, d + 1]

    lhs = torch.cat((left_block, right_block), dim=2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros(batch_size, d + 1, k, dtype=train_points.dtype, device=device).float()
    rhs = torch.cat((f, rhs_zeros), dim=1)  # [b, n + d + 1, k]

    X = torch.linalg.solve(lhs, rhs)
    w = X[:, :n, :]
    v = X[:, n:, :]
    return w, v


def cross_squared_distance_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise squared distances between rows of x and y.

    Args:
        x: Tensor of shape [batch_size, n, d].
        y: Tensor of shape [batch_size, m, d].

    Returns:
        Tensor of shape [batch_size, n, m] of squared distances.
    """
    batch_size, n, d = x.shape
    _, m, _ = y.shape

    x_expanded = x.unsqueeze(2)  # [b, n, 1, d]
    y_expanded = y.unsqueeze(1)  # [b, 1, m, d]
    diffs = x_expanded - y_expanded  # [b, n, m, d]
    squared_dists = torch.sum(diffs * diffs, dim=3)  # [b, n, m]

    return squared_dists


def phi(r: torch.Tensor, order: int) -> torch.Tensor:
    """
    Coordinate-wise nonlinearity defining the order of interpolation.

    Args:
        r: Input tensor.
        order: Interpolation order.

    Returns:
        Tensor after applying phi function coordinate-wise.
    """
    EPSILON = torch.tensor(1e-10, device=r.device)
    if order == 1:
        r = torch.max(r, EPSILON)
        return torch.sqrt(r)
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def apply_interpolation(
    query_points: torch.Tensor,
    train_points: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """
    Evaluate polyharmonic spline interpolation at query points.

    Args:
        query_points: Tensor of shape [m, d].
        train_points: Tensor of shape [batch_size, n, d].
        w: Tensor of shape [batch_size, n, k].
        v: Tensor of shape [batch_size, d + 1, k].
        order: Interpolation order.

    Returns:
        Tensor of shape [batch_size, m, k] of interpolated values.
    """
    batch_size, n, d = train_points.shape
    k = w.shape[-1]
    m = query_points.shape[0]

    query_points_expanded = query_points.unsqueeze(0).expand(batch_size, m, d)  # [b, m, d]

    pairwise_dists = cross_squared_distance_matrix(query_points_expanded.float(), train_points.float())
    phi_pairwise_dists = phi(pairwise_dists, order)  # [b, m, n]

    rbf_term = torch.bmm(phi_pairwise_dists, w)  # [b, m, k]

    ones = torch.ones(batch_size, m, 1, device=query_points.device, dtype=query_points.dtype)
    query_points_pad = torch.cat((query_points_expanded, ones), dim=2).float()  # [b, m, d + 1]

    linear_term = torch.bmm(query_points_pad, v)  # [b, m, k]

    return rbf_term + linear_term


def dense_image_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    Warp image using dense flow vectors with bilinear interpolation.

    Args:
        image: Tensor of shape [batch_size, height, width].
        flow: Tensor of shape [batch_size, height, width, 2].

    Returns:
        Warped image tensor of shape [batch_size, height, width].
    """
    # Add channel dimension
    image = image.unsqueeze(3)  # [b, h, w, 1]
    batch_size, height, width, channels = image.shape
    device = image.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'
    )  # [h, w]

    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()  # [h, w, 2]

    batched_grid = stacked_grid.unsqueeze(0).expand(batch_size, height, width, 2)  # [b, h, w, 2]

    query_points_on_grid = batched_grid - flow  # [b, h, w, 2]
    query_points_flattened = query_points_on_grid.reshape(batch_size, height * width, 2)

    interpolated = interpolate_bilinear(image, query_points_flattened)

    interpolated = interpolated.reshape(batch_size, height, width, channels)

    return interpolated.squeeze(3)  # remove channel dimension


def interpolate_bilinear(
    grid: torch.Tensor,
    query_points: torch.Tensor,
    indexing: str = 'ij',
) -> torch.Tensor:
    """
    Bilinear interpolation of values at query points on a grid.

    Args:
        grid: Tensor of shape [batch_size, height, width, channels].
        query_points: Tensor of shape [batch_size, num_queries, 2].
        indexing: 'ij' for row-column indexing, 'xy' for Cartesian.

    Returns:
        Tensor of shape [batch_size, num_queries, channels].
    """
    if indexing not in ('ij', 'xy'):
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    batch_size, height, width, channels = grid.shape
    num_queries = query_points.shape[1]
    device = grid.device
    dtype = grid.dtype

    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = query_points.unbind(dim=2)

    alphas = []
    floors = []
    ceils = []

    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_dim = height if dim == 0 else width

        max_floor = size_in_dim - 2
        floor = torch.floor(queries).clamp(min=0, max=max_floor).long()
        ceil = floor + 1

        floors.append(floor)
        ceils.append(ceil)

        alpha = (queries - floor.float()).clamp(min=0.0, max=1.0).unsqueeze(2).to(dtype)
        alphas.append(alpha)

    flattened_grid = grid.reshape(batch_size * height * width, channels)
    batch_offsets = (torch.arange(batch_size, device=device) * height * width).unsqueeze(1)

    def gather(y_coords: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        linear_indices = batch_offsets + y_coords * width + x_coords
        gathered = torch.gather(flattened_grid.t(), 1, linear_indices)
        return gathered.reshape(batch_size, num_queries, channels)

    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp