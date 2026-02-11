import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def visualize_anchors(info_path, config_path, save_path):
    # Load VoD Infos
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    # Load Model Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    anchor_configs = config['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG']
    
    # Extract GT sizes
    gt_sizes = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    for info in infos:
        annos = info['annos']
        names = annos['name']
        dims = annos['gt_boxes_lidar'][:, 3:5] # l, w (dx, dy)
        
        for name, dim in zip(names, dims):
            if name in gt_sizes:
                gt_sizes[name].append(dim)
    
    plt.figure(figsize=(10, 8))
    colors = {'Car': 'green', 'Pedestrian': 'red', 'Cyclist': 'blue'}
    
    for cls, sizes in gt_sizes.items():
        if not sizes: continue
        sizes = np.array(sizes)
        plt.scatter(sizes[:, 0], sizes[:, 1], s=5, alpha=0.3, label=f'{cls} GT', color=colors[cls])
        
        # Find corresponding anchor size
        for anchor in anchor_configs:
            if anchor['class_name'] == cls:
                # anchor_sizes are [[l, w, h]]
                a_l, a_w = anchor['anchor_sizes'][0][0], anchor['anchor_sizes'][0][1]
                plt.scatter(a_l, a_w, s=200, marker='X', edgecolors='black', label=f'{cls} Anchor ({a_l}x{a_w})', color=colors[cls])

    plt.xlabel('Length (L / dx) [m]')
    plt.ylabel('Width (W / dy) [m]')
    plt.title('VoD GT Sizes vs Configured Anchors')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f'Saved anchor verification plot to {save_path}')

if __name__ == '__main__':
    info_path = Path('data/VoD/view_of_delft_PUBLIC/radar_5frames/vod_infos_train.pkl')
    config_path = Path('tools/cfgs/vod_models/vod_radarpillar.yaml')
    save_path = Path('tools/anchor_verification.png')
    
    if not info_path.exists():
        print(f'Error: {info_path} not found.')
    elif not config_path.exists():
        print(f'Error: {config_path} not found.')
    else:
        visualize_anchors(info_path, config_path, save_path)
