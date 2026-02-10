import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSOLUTE_XYZ', self.model_cfg.get('USE_ABSLOTE_XYZ'))
        self.use_velocity_decomposition = self.model_cfg.get('USE_VELOCITY_DECOMPOSITION', False)
        self.use_rel_velocity_decomposition = self.model_cfg.get('USE_REL_VELOCITY_DECOMPOSITION', False)
        self.use_velocity_offset = self.model_cfg.get('USE_VELOCITY_OFFSET', False)
        self.use_rel_velocity_offset = self.model_cfg.get('USE_REL_VELOCITY_OFFSET', False)
        self.velocity_comp_index = self.model_cfg.get('VELOCITY_COMP_INDEX', 4)
        self.velocity_rel_index = self.model_cfg.get('VELOCITY_REL_INDEX', None)
        self.feature_order = self.model_cfg.get('FEATURE_ORDER', None)
        self.rcs_index = self.model_cfg.get('RCS_INDEX', 3)
        self.time_index = self.model_cfg.get('TIME_INDEX', None)
        self.normalize_velocity_comp = self.model_cfg.get('NORMALIZE_VELOCITY_COMP', False)
        self.normalize_velocity_rel = self.model_cfg.get('NORMALIZE_VELOCITY_REL', False)
        if self.normalize_velocity_comp:
            velocity_comp_mean = self.model_cfg.get('VELOCITY_COMP_MEAN', None)
            velocity_comp_std = self.model_cfg.get('VELOCITY_COMP_STD', None)
            if velocity_comp_mean is None or velocity_comp_std is None:
                raise ValueError('VELOCITY_COMP_MEAN/STD must be set when NORMALIZE_VELOCITY_COMP is True')
            if len(velocity_comp_mean) != 2 or len(velocity_comp_std) != 2:
                raise ValueError('VELOCITY_COMP_MEAN/STD must be length 2 for [vx, vy]')
            self.register_buffer(
                'velocity_comp_mean',
                torch.tensor(velocity_comp_mean, dtype=torch.float32).view(1, 1, 2)
            )
            self.register_buffer(
                'velocity_comp_std',
                torch.tensor(velocity_comp_std, dtype=torch.float32).view(1, 1, 2)
            )
        if self.normalize_velocity_rel:
            velocity_rel_mean = self.model_cfg.get('VELOCITY_REL_MEAN', None)
            velocity_rel_std = self.model_cfg.get('VELOCITY_REL_STD', None)
            if velocity_rel_mean is None or velocity_rel_std is None:
                raise ValueError('VELOCITY_REL_MEAN/STD must be set when NORMALIZE_VELOCITY_REL is True')
            if len(velocity_rel_mean) != 2 or len(velocity_rel_std) != 2:
                raise ValueError('VELOCITY_REL_MEAN/STD must be length 2 for [vx, vy]')
            self.register_buffer(
                'velocity_rel_mean',
                torch.tensor(velocity_rel_mean, dtype=torch.float32).view(1, 1, 2)
            )
            self.register_buffer(
                'velocity_rel_std',
                torch.tensor(velocity_rel_std, dtype=torch.float32).view(1, 1, 2)
            )
        if (self.use_rel_velocity_decomposition or self.use_rel_velocity_offset) and self.velocity_rel_index is None:
            raise ValueError('VELOCITY_REL_INDEX must be set when using relative velocity features')
        if self.feature_order is not None:
            if not isinstance(self.feature_order, (list, tuple)):
                raise ValueError('FEATURE_ORDER must be a list of feature names')
            if any(name in ['vx', 'vy'] for name in self.feature_order) and not self.use_velocity_decomposition:
                raise ValueError('USE_VELOCITY_DECOMPOSITION must be True when FEATURE_ORDER includes vx/vy')
            if any(name in ['vx_rel', 'vy_rel'] for name in self.feature_order) and not self.use_rel_velocity_decomposition:
                raise ValueError('USE_REL_VELOCITY_DECOMPOSITION must be True when FEATURE_ORDER includes vx_rel/vy_rel')
            if 'v_offset' in self.feature_order and not self.use_velocity_offset:
                raise ValueError('USE_VELOCITY_OFFSET must be True when FEATURE_ORDER includes v_offset')
            if 'v_rel_offset' in self.feature_order and not self.use_rel_velocity_offset:
                raise ValueError('USE_REL_VELOCITY_OFFSET must be True when FEATURE_ORDER includes v_rel_offset')
            if 'time' in self.feature_order and self.time_index is None:
                raise ValueError('TIME_INDEX must be set when FEATURE_ORDER includes time')
            if 'rcs' in self.feature_order and self.rcs_index is None:
                raise ValueError('RCS_INDEX must be set when FEATURE_ORDER includes rcs')
            if (not self.use_absolute_xyz) and self.feature_order[:3] != ['x', 'y', 'z']:
                raise ValueError('FEATURE_ORDER must start with x,y,z when USE_ABSOLUTE_XYZ is False')
            num_point_features = len(self.feature_order)
        else:
            if self.use_velocity_decomposition:
                num_point_features += 2  # vx, vy
            if self.use_rel_velocity_decomposition:
                num_point_features += 2  # vx_rel, vy_rel
            if self.use_velocity_offset:
                num_point_features += 1  # vr_offset (vr,m)
            if self.use_rel_velocity_offset:
                num_point_features += 1  # vrel_offset (vrel,m)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        base_features = voxel_features
        phi = torch.atan2(base_features[:, :, 1], base_features[:, :, 0] + 1e-6)
        velocity_rel = None
        velocity_comp = None
        v_rel = None
        v_comp = None
        v_rel_offset = None
        v_comp_offset = None

        if self.use_rel_velocity_decomposition:
            v_rel = base_features[:, :, self.velocity_rel_index]
            vx_rel = v_rel * torch.cos(phi)
            vy_rel = v_rel * torch.sin(phi)
            velocity_rel = torch.stack([vx_rel, vy_rel], dim=-1)
            if self.normalize_velocity_rel:
                mean = self.velocity_rel_mean.type_as(velocity_rel)
                std = self.velocity_rel_std.type_as(velocity_rel).clamp(min=1e-6)
                velocity_rel = (velocity_rel - mean) / std
            if self.feature_order is None:
                voxel_features = torch.cat([voxel_features, velocity_rel], dim=-1)

        if self.use_velocity_decomposition:
            v_comp = base_features[:, :, self.velocity_comp_index]
            vx_comp = v_comp * torch.cos(phi)
            vy_comp = v_comp * torch.sin(phi)
            velocity_comp = torch.stack([vx_comp, vy_comp], dim=-1)
            if self.normalize_velocity_comp:
                # (1, 1, 2) stats broadcast over (num_voxels, num_points, 2)
                mean = self.velocity_comp_mean.type_as(velocity_comp)
                std = self.velocity_comp_std.type_as(velocity_comp).clamp(min=1e-6)
                velocity_comp = (velocity_comp - mean) / std
            if self.feature_order is None:
                voxel_features = torch.cat([voxel_features, velocity_comp], dim=-1)

        if self.use_rel_velocity_offset or self.use_velocity_offset:
            mask = self.get_paddings_indicator(voxel_num_points, base_features.shape[1], axis=0)
            mask = mask.float()

        if self.use_rel_velocity_offset:
            if v_rel is None:
                v_rel = base_features[:, :, self.velocity_rel_index]
            v_rel_masked = v_rel * mask
            v_rel_mean = v_rel_masked.sum(dim=1, keepdim=True) / voxel_num_points.float().unsqueeze(1).clamp(min=1)
            v_rel_offset = v_rel - v_rel_mean
            if self.feature_order is None:
                voxel_features = torch.cat([voxel_features, v_rel_offset.unsqueeze(-1)], dim=-1)

        if self.use_velocity_offset:
            # Calculate offset velocity (vr,m) as described in RadarPillars paper
            if v_comp is None:
                v_comp = base_features[:, :, self.velocity_comp_index]
            v_comp_masked = v_comp * mask
            v_comp_mean = v_comp_masked.sum(dim=1, keepdim=True) / voxel_num_points.float().unsqueeze(1).clamp(min=1)
            v_comp_offset = v_comp - v_comp_mean
            if self.feature_order is None:
                voxel_features = torch.cat([voxel_features, v_comp_offset.unsqueeze(-1)], dim=-1)

        if self.feature_order is not None:
            ordered_features = []
            for name in self.feature_order:
                if name == 'x':
                    ordered_features.append(base_features[:, :, 0:1])
                elif name == 'y':
                    ordered_features.append(base_features[:, :, 1:2])
                elif name == 'z':
                    ordered_features.append(base_features[:, :, 2:3])
                elif name == 'rcs':
                    ordered_features.append(base_features[:, :, self.rcs_index:self.rcs_index + 1])
                elif name == 'time':
                    ordered_features.append(base_features[:, :, self.time_index:self.time_index + 1])
                elif name == 'v_r':
                    ordered_features.append(base_features[:, :, self.velocity_rel_index:self.velocity_rel_index + 1])
                elif name == 'v_r_comp':
                    ordered_features.append(base_features[:, :, self.velocity_comp_index:self.velocity_comp_index + 1])
                elif name == 'vx':
                    if velocity_comp is None:
                        raise ValueError('Velocity decomposition is required for vx')
                    ordered_features.append(velocity_comp[:, :, 0:1])
                elif name == 'vy':
                    if velocity_comp is None:
                        raise ValueError('Velocity decomposition is required for vy')
                    ordered_features.append(velocity_comp[:, :, 1:2])
                elif name == 'vx_rel':
                    if velocity_rel is None:
                        raise ValueError('Relative velocity decomposition is required for vx_rel')
                    ordered_features.append(velocity_rel[:, :, 0:1])
                elif name == 'vy_rel':
                    if velocity_rel is None:
                        raise ValueError('Relative velocity decomposition is required for vy_rel')
                    ordered_features.append(velocity_rel[:, :, 1:2])
                elif name == 'v_offset':
                    if v_comp_offset is None:
                        raise ValueError('Velocity offset is required for v_offset')
                    ordered_features.append(v_comp_offset.unsqueeze(-1))
                elif name == 'v_rel_offset':
                    if v_rel_offset is None:
                        raise ValueError('Relative velocity offset is required for v_rel_offset')
                    ordered_features.append(v_rel_offset.unsqueeze(-1))
                else:
                    raise ValueError(f'Unsupported FEATURE_ORDER entry: {name}')
            voxel_features = torch.cat(ordered_features, dim=-1)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
