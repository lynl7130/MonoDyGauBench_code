from .create_from_pcd import create_from_pcd_D3G
from typing import Callable, List
from src.utils.graphics_utils import BasicPointCloud

def create_from_pcd_func(
    pcd: BasicPointCloud, spatial_lr_scale: float,
    max_sh_degree: int,
    init_mode: str) -> List:
    if init_mode == "D3G":
        return create_from_pcd_D3G(pcd, spatial_lr_scale, max_sh_degree)
    else:
        assert False, f"Unrecognizable create_from_pcd mode: {init_mode}"