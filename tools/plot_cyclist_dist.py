import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cyclist_dist(info_path, save_path):
    # Load VoD Infos
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    # Extract Cyclist lengths
    cyclist_lengths = []
    for info in infos:
        annos = info['annos']
        names = annos['name']
        dims = annos['gt_boxes_lidar'][:, 3] # length (dx)
        
        for name, length in zip(names, dims):
            if name == 'Cyclist':
                cyclist_lengths.append(length)
    
    if not cyclist_lengths:
        print('No Cyclists found in the info file.')
        return

    plt.figure(figsize=(10, 6))
    plt.hist(cyclist_lengths, bins=50, color='blue', edgecolor='black', alpha=0.7)
    
    # Calculate statistics
    avg_len = np.mean(cyclist_lengths)
    median_len = np.median(cyclist_lengths)
    
    plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_len:.2f}m')
    plt.axvline(median_len, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len:.2f}m')
    
    plt.xlabel('Length (dx) [m]')
    plt.ylabel('Frequency')
    plt.title('VoD Cyclist Length Distribution')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    print(f'Saved cyclist distribution plot to {save_path}')
    
    # Print summary
    print(f'Total Cyclists: {len(cyclist_lengths)}')
    print(f'Min Length: {min(cyclist_lengths):.2f}m')
    print(f'Max Length: {max(cyclist_lengths):.2f}m')
    print(f'Average Length: {avg_len:.2f}m')

if __name__ == '__main__':
    info_path = Path('data/VoD/view_of_delft_PUBLIC/radar_5frames/vod_infos_train.pkl')
    save_path = Path('tools/cyclist_dist.png')
    
    if not info_path.exists():
        print(f'Error: {info_path} not found.')
    else:
        plot_cyclist_dist(info_path, save_path)
