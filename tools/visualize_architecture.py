
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path=None, logger=None, num_points=2048):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,  # Avoid DB sampler dependency from train augmentor.
            root_path=root_path,
            logger=logger,
        )
        self.num_points = num_points
        self.src_feature_list = list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        self.num_src_features = len(self.src_feature_list)

        self.dummy_points = self._build_dummy_points()
        self.dummy_gt_boxes, self.dummy_gt_names = self._build_dummy_gt_boxes()

    def _build_dummy_points(self):
        points = np.zeros((self.num_points, self.num_src_features), dtype=np.float32)
        pc_range = self.point_cloud_range

        points[:, 0] = np.random.uniform(pc_range[0] + 0.1, pc_range[3] - 0.1, self.num_points)
        points[:, 1] = np.random.uniform(pc_range[1] + 0.1, pc_range[4] - 0.1, self.num_points)
        points[:, 2] = np.random.uniform(pc_range[2] + 0.1, pc_range[5] - 0.1, self.num_points)

        for feat_idx, feat_name in enumerate(self.src_feature_list[3:], start=3):
            if feat_name == 'rcs':
                points[:, feat_idx] = np.random.normal(loc=-10.0, scale=6.0, size=self.num_points)
            elif feat_name in ['vr', 'v_r', 'v_r_comp']:
                points[:, feat_idx] = np.random.normal(loc=0.0, scale=2.0, size=self.num_points)
            elif feat_name == 'time':
                points[:, feat_idx] = np.random.uniform(-2.5, 2.5, self.num_points)
            else:
                points[:, feat_idx] = np.random.normal(loc=0.0, scale=1.0, size=self.num_points)

        return points.astype(np.float32)

    def _build_dummy_gt_boxes(self):
        if len(self.class_names) == 0:
            raise ValueError('CLASS_NAMES must contain at least one class for dummy gt_boxes.')

        default_size_by_class = {
            'Car': (3.9, 1.6, 1.56),
            'Pedestrian': (0.8, 0.6, 1.73),
            'Cyclist': (1.76, 0.6, 1.73),
        }

        num_boxes = min(2, len(self.class_names))
        gt_boxes = np.zeros((num_boxes, 7), dtype=np.float32)
        gt_names = []

        pc_range = self.point_cloud_range
        for box_idx in range(num_boxes):
            class_name = self.class_names[box_idx]
            dx, dy, dz = default_size_by_class.get(class_name, (2.0, 1.0, 1.5))

            gt_boxes[box_idx, 0] = np.random.uniform(pc_range[0] + 5.0, pc_range[3] - 5.0)  # x
            gt_boxes[box_idx, 1] = np.random.uniform(pc_range[1] + 5.0, pc_range[4] - 5.0)  # y
            gt_boxes[box_idx, 2] = np.random.uniform(-1.0, 1.0)  # z
            gt_boxes[box_idx, 3] = dx
            gt_boxes[box_idx, 4] = dy
            gt_boxes[box_idx, 5] = dz
            gt_boxes[box_idx, 6] = np.random.uniform(-np.pi, np.pi)  # heading
            gt_names.append(class_name)

        return gt_boxes, np.array(gt_names)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        input_dict = {
            'points': self.dummy_points.copy(),
            'frame_id': f'demo_{index:06d}',
            'gt_boxes': self.dummy_gt_boxes.copy(),
            'gt_names': self.dummy_gt_names.copy(),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class BevBackboneHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone_2d = model.backbone_2d
        self.conv_cls = model.dense_head.conv_cls
        self.conv_box = model.dense_head.conv_box
        self.conv_dir_cls = model.dense_head.conv_dir_cls

    def forward(self, spatial_features):
        # spatial_features: (B, C_bev, H_bev, W_bev)
        batch_dict = {'spatial_features': spatial_features}
        batch_dict = self.backbone_2d(batch_dict)

        # spatial_features_2d: (B, C_2d, H_2d, W_2d)
        spatial_features_2d = batch_dict['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box_preds = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
            return cls_preds, box_preds, dir_cls_preds
        return cls_preds, box_preds


def parse_config():
    parser = argparse.ArgumentParser(description='Visualize RadarPillars model architecture')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='tools/cfgs/dataset_configs/vod_dataset_radar.yaml',
        help='Config path for visualization run',
    )
    parser.add_argument('--output_dir', type=str, default='output/architecture_viz', help='Output directory')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of dummy points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip_onnx', action='store_true', help='Skip ONNX export stage')
    parser.add_argument('--cpu', action='store_true', help='(Not supported for this model) Keep for compatibility')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def clone_batch_dict(batch_dict):
    copied = {}
    for key, val in batch_dict.items():
        if torch.is_tensor(val):
            copied[key] = val.clone()
        elif isinstance(val, np.ndarray):
            copied[key] = val.copy()
        else:
            copied[key] = copy.deepcopy(val)
    return copied


def load_batch_to_device(batch_dict, device):
    for key, val in batch_dict.items():
        if isinstance(val, np.ndarray):
            if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
                continue
            tensor = torch.from_numpy(val)
            if key in ['voxel_coords']:
                batch_dict[key] = tensor.to(device=device, dtype=torch.int32)
            else:
                batch_dict[key] = tensor.to(device=device, dtype=torch.float32)
        elif torch.is_tensor(val):
            batch_dict[key] = val.to(device)

    return batch_dict


def save_torchviz_graph(model, batch_dict, output_dir, logger):
    try:
        import torchviz
    except ImportError as err:
        raise RuntimeError('torchviz is not installed. Install with: pip install torchviz') from err

    model.train()
    batch_for_viz = clone_batch_dict(batch_dict)
    ret_dict, _, _ = model(batch_for_viz)
    loss = ret_dict['loss']

    dot = torchviz.make_dot(loss, params=dict(model.named_parameters()))
    dot_file = output_dir / 'model_architecture_torchviz.dot'
    png_prefix = output_dir / 'model_architecture_torchviz'
    dot.save(str(dot_file))

    try:
        dot.render(str(png_prefix), format='png', cleanup=True)
        logger.info(f'Torchviz PNG saved: {png_prefix}.png')
    except Exception as err:  # noqa: BLE001
        logger.warning(f'Graphviz render failed ({err}). DOT file is still available at: {dot_file}')


def get_spatial_features(model, batch_dict):
    model.eval()
    batch_for_onnx = clone_batch_dict(batch_dict)
    with torch.no_grad():
        for module_name in ['vfe', 'backbone_3d', 'map_to_bev_module']:
            module = getattr(model, module_name, None)
            if module is not None:
                batch_for_onnx = module(batch_for_onnx)

    if 'spatial_features' not in batch_for_onnx:
        raise RuntimeError('spatial_features was not generated by VFE/backbone/map_to_bev pipeline.')
    return batch_for_onnx['spatial_features']


def export_onnx_bev_head(model, spatial_features, output_dir, logger):
    wrapper = BevBackboneHeadWrapper(model).to(spatial_features.device)
    wrapper.eval()

    output_path = output_dir / 'model_architecture_bev_head.onnx'
    output_names = ['cls_preds', 'box_preds']
    if wrapper.conv_dir_cls is not None:
        output_names.append('dir_cls_preds')

    dynamic_axes = {
        'spatial_features': {0: 'batch_size'},
        'cls_preds': {0: 'batch_size'},
        'box_preds': {0: 'batch_size'},
    }
    if 'dir_cls_preds' in output_names:
        dynamic_axes['dir_cls_preds'] = {0: 'batch_size'}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            spatial_features,
            str(output_path),
            input_names=['spatial_features'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    logger.info(f'ONNX export saved: {output_path}')


def main():
    args, cfg = parse_config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = common_utils.create_logger()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        logger.warning('CPU mode was requested, but this model path depends on CUDA anchor generation.')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA GPU is required for this visualization script in this repo '
            '(anchor generator uses CUDA-only tensor creation).'
        )

    device = torch.device('cuda')
    logger.info('Using device: cuda')

    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path('/tmp'),
        logger=logger,
        num_points=args.num_points,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).to(device)

    data_dict = dataset[0]
    data_dict = dataset.collate_batch([data_dict])
    data_dict = load_batch_to_device(data_dict, device)

    logger.info('=' * 60)
    logger.info('GENERATING MODEL VISUALIZATIONS')
    logger.info('=' * 60)

    save_torchviz_graph(model=model, batch_dict=data_dict, output_dir=output_dir, logger=logger)

    if not args.skip_onnx:
        try:
            spatial_features = get_spatial_features(model=model, batch_dict=data_dict)
            export_onnx_bev_head(
                model=model,
                spatial_features=spatial_features,
                output_dir=output_dir,
                logger=logger,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning(f'ONNX export failed: {err}')


if __name__ == '__main__':
    main()
