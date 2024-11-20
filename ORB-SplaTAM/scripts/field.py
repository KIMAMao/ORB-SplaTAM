import torch
from collections import defaultdict


class Field:
    def __init__(self, device='cpu'):
        self._device = device
        self._init_map_dict()

    def _init_map_dict(self):
        """初始化 self._global_map_dict 以及相关属性."""
        self._global_map_dict = {
            "positions": torch.zeros(32, 3, device=self._device),
            "orientations": torch.zeros(32, 4, device=self._device),
            "kf_ids": torch.zeros(32, dtype=torch.long, device=self._device),
            "training_iterations": torch.zeros(32, dtype=torch.long, device=self._device),
            "num": 0,
        }
        self._kf2fields = defaultdict(set)

    @utils.benchmark
    @torch.no_grad()
    def _extend_global_map_dict(
            self,
            depth_image: torch.Tensor,
            frame_id: int,
            c2w: torch.Tensor,
            camera: camera.Camera,
            active_map_dict: Optional[dict] = None,
    ) -> dict:
        """Ensure fields cover depth image or add new fields."""
        logging.debug("NeuralGraphMap._extend_global_map_dict")

        # 点云转换
        xyz_cam = camera.depth_to_pointcloud(depth_image, convention="opengl")
        xyz_world = utils.transform_points(xyz_cam, c2w)

        # 更新包围盒
        bb_max = xyz_world.max(dim=0)[0]
        bb_min = xyz_world.max(dim=0)[0]

        self._bb_max = torch.where(bb_max > self._bb_max, bb_max, self._bb_max)
        self._bb_min = torch.where(bb_min < self._bb_min, bb_min, self._bb_min)

        # check uncovered points exactly
        if active_map_dict is not None:
            _, idx, _ = ball_query(
                xyz_world.unsqueeze(0),
                active_map_dict["positions"].unsqueeze(0),
                K=1,
                radius=self._field_radius,
                return_nn=False,
            )
            xyz_world = xyz_world[idx[0, :, 0] == -1]

        # use grid to cover these points
        cell_size = 2 * self._field_radius / math.sqrt(3)  # 定义网格的单元尺寸
        shift = torch.empty((3,), device=self._device).uniform_(0.0, cell_size)  # 随机平移网格
        to_be_covered_ijk = ((xyz_world + shift) / cell_size).floor().unique(dim=0)  # 得到每个点云位置对应的网格索引

        if active_map_dict is not None:
            global_field_positions = active_map_dict["positions"]  # 获取当前活跃地图所有的点云位置
            covered_ijk = ((global_field_positions + shift) / cell_size).floor().unique(dim=0)  # 得到上面所有点的网格索引
        else:
            covered_ijk = torch.empty(0, 3, device=self._device)

        combined_ijk = torch.cat((to_be_covered_ijk, covered_ijk))  # 将待覆盖的网格索引和已覆盖的网格索引合并
        _, inv, counts = combined_ijk.unique(dim=0, return_inverse=True, return_counts=True)  # 去除重复的网格索引
        new_ijk = to_be_covered_ijk[counts[inv[: len(to_be_covered_ijk)]] == 1]  # 确定新的未覆盖的网格索引

        num_prev = self._global_map_dict["num"]
        num_new = len(new_ijk)

        # 没有新的uncover的点
        if num_new == 0:
            return

        if (
                self._global_map_dict["positions"].shape[0]
                <= self._global_map_dict["num"] + num_new
        ):
            self._extend_map_dict(self._global_map_dict["num"] + len(new_ijk))

        cell_center = (new_ijk - shift + 0.5) * cell_size  # 每个uncover点的场的中心

        # 新加入的点，在_global_map_dict这个字典里的起始位置
        start = self._global_map_dict["num"]
        end = self._global_map_dict["num"] + num_new

        self._global_map_dict["positions"][start:end] = cell_center
        self._global_map_dict["orientations"][start:end] = torch.zeros(
            num_new, 4, device=self._device
        )
        self._global_map_dict["orientations"][start:end, 0] = 1.0
        self._global_map_dict["kf_ids"][start:end] = frame_id  # 记录这些新网格单元（新加入的uncover的点）对应的关键帧ID
        self._global_map_dict["num"] += num_new
        self._global_map_dict["training_iterations"][start:end] = torch.zeros(
            num_new, device=self._device, dtype=torch.long
        )  # 为新网格单元设置训练迭代次数，初始化为0

        self._add_fields(num_new)

        self._kf2fields[frame_id] = {
            field_id for field_id in range(num_prev, self._global_map_dict["num"])
        }  # 这个字典用来记录每个关键帧ID (frame_id) 与其对应的网格单元ID的关联

def _extend_global_field_dict(global_field_dict, new_pt_cld, frame_id, device='cuda'):
    """Ensure fields cover new point cloud or add new fields."""
    # Define field radius and cell size for grid
    field_radius = 1.0  # Assume a predefined field radius
    cell_size = 2 * field_radius / math.sqrt(3)  # Define grid cell size

    # Convert points to grid indices
    shift = torch.empty((3,), device=device).uniform_(0.0, cell_size)
    grid_indices = ((new_pt_cld[:, :3] + shift) / cell_size).floor().int()

    # Unique grid cells
    unique_indices, inverse_indices = torch.unique(grid_indices, dim=0, return_inverse=True)

    # Check for uncovered points
    if 'positions' in global_field_dict and global_field_dict['num'] > 0:
        existing_indices = ((global_field_dict['positions'] + shift) / cell_size).floor().int()
        existing_unique_indices = torch.unique(existing_indices, dim=0)
        new_unique_indices = [idx for idx in unique_indices if idx not in existing_unique_indices]
    else:
        new_unique_indices = unique_indices

    # Calculate new field centers
    new_centers = (new_unique_indices.float() + 0.5) * cell_size - shift

    # Add new fields to the dictionary
    num_new_fields = new_centers.shape[0]
    if num_new_fields > 0:
        start = global_field_dict['num']
        end = start + num_new_fields

        # Resize tensor arrays if necessary
        if end > global_field_dict['positions'].shape[0]:
            extra_size = end - global_field_dict['positions'].shape[0]
            global_field_dict['positions'] = torch.cat([global_field_dict['positions'], torch.zeros(extra_size, 3, device=device)])
            global_field_dict['orientations'] = torch.cat([global_field_dict['orientations'], torch.zeros(extra_size, 4, device=device)])
            global_field_dict['kf_ids'] = torch.cat([global_field_dict['kf_ids'], torch.zeros(extra_size, dtype=torch.long, device=device)])
            global_field_dict['orientations'][:, 0] = 1  # Default quaternion (no rotation)

        global_field_dict['positions'][start:end] = new_centers
        global_field_dict['kf_ids'][start:end] = frame_id
        global_field_dict['num'] += num_new_fields

    return global_field_dict

map_manager = Field(device='cuda')


