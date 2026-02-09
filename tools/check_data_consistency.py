import pickle
from pathlib import Path
import tqdm
from collections import Counter

def check_data_consistency(info_path, label_dir):
    # Load VoD Infos
    print(f'Loading info file: {info_path}')
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    label_dir = Path(label_dir)
    cyclist_mismatches = 0
    total_checked = 0
    all_raw_classes = Counter()
    
    # Class mappings for Cyclist from config
    cyclist_map = ['bicycle', 'rider', 'Cyclist', 'moped_scooter', 'motor', 'ride_other', 'ride_uncertain']
    
    print('Checking consistency between PKL and raw labels (Focus: Cyclist)...')
    for info in tqdm.tqdm(infos):
        frame_id = info['point_cloud']['lidar_idx']
        label_file = label_dir / f'{frame_id}.txt'
        
        if not label_file.exists():
            continue
            
        # Count objects in raw label
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        raw_names = [l.split()[0] for l in lines]
        all_raw_classes.update(raw_names)
        
        raw_cyclist_count = sum(1 for name in raw_names if name in cyclist_map)
        
        # Count objects in info (annos['name'])
        info_cyclist_count = sum(1 for name in info['annos']['name'] if name == 'Cyclist')
        
        if raw_cyclist_count != info_cyclist_count:
            print(f'\n[CYCLIST MISMATCH] Frame {frame_id}: Raw has {raw_cyclist_count}, but info has {info_cyclist_count}.')
            cyclist_mismatches += 1
        
        total_checked += 1

    print('\n' + '='*30)
    print(f'Total frames checked: {total_checked}')
    print(f'Total Cyclist inconsistencies: {cyclist_mismatches}')
    print('\nRaw Class Distribution:')
    for cls, count in all_raw_classes.most_common():
        print(f'  {cls}: {count}')
    print('='*30)

if __name__ == '__main__':
    info_path = Path('data/VoD/view_of_delft_PUBLIC/radar_5frames/vod_infos_train.pkl')
    label_dir = Path('data/VoD/view_of_delft_PUBLIC/radar_5frames/training/label_2')
    
    if not info_path.exists():
        print(f'Error: {info_path} not found.')
    elif not label_dir.exists():
        print(f'Error: {label_dir} not found.')
    else:
        check_data_consistency(info_path, label_dir)
