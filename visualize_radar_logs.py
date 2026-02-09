import re
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def parse_logs(log_paths):
    data = {}
    
    # regex patterns
    epoch_pattern = re.compile(r"\*+ EPOCH (\d+) EVALUATION \*+")
    class_pattern = re.compile(r"(Car|Pedestrian|Cyclist) AP_R40@.*")
    metric_pattern = re.compile(r"(bbox|bev|3d)\s+AP:([\d\.]+), ([\d\.]+), ([\d\.]+)")
    
    current_epoch = None
    current_class = None
    
    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found.")
            continue
            
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_class = None # Reset class for new epoch
                data.setdefault(current_epoch, {})
                continue
                
            if current_epoch is not None:
                class_match = class_pattern.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    data[current_epoch].setdefault(current_class, {})
                    continue
                
                if current_class is not None:
                    metric_match = metric_pattern.search(line)
                    if metric_match:
                        m_type = metric_match.group(1)
                        val = float(metric_match.group(2))
                        # Ensure class exists in current epoch data
                        data[current_epoch].setdefault(current_class, {})[m_type] = val
                        
    return data

def plot_metrics(data, output_dir):
    if not data:
        print("No data found to plot.")
        return

    epochs = sorted(data.keys())
    classes = ['Car', 'Pedestrian', 'Cyclist']
    metrics = ['bbox', 'bev', '3d']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Individual Class Plots (3D AP)
    plt.figure(figsize=(10, 6))
    for cls in classes:
        values = [data[e].get(cls, {}).get('3d', np.nan) for e in epochs]
        plt.plot(epochs, values, marker='o', label=cls)
    
    plt.title('RadarPillar 3D AP (R40) Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('3D AP')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '3d_ap_evolution.png'))
    plt.close()
    
    # 2. mAP (Mean AP across classes for each metric)
    plt.figure(figsize=(10, 6))
    for m in metrics:
        m_values = []
        for e in epochs:
            cls_vals = [data[e].get(cls, {}).get(m, np.nan) for cls in classes]
            m_values.append(np.nanmean(cls_vals) if not all(np.isnan(cls_vals)) else np.nan)
        plt.plot(epochs, m_values, marker='s', label=f'mAP {m}')
        
    plt.title('RadarPillar Mean AP Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'map_evolution.png'))
    plt.close()

    # 3. Detailed per-class plots (BBox vs BEV vs 3D)
    for cls in classes:
        plt.figure(figsize=(10, 6))
        for m in metrics:
            values = [data[e].get(cls, {}).get(m, np.nan) for e in epochs]
            plt.plot(epochs, values, marker='o', label=f'{cls} {m}')
        plt.title(f'{cls} AP Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{cls.lower()}_ap_evolution.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="List of log files to parse")
    parser.add_argument("--output", default="output_plots", help="Directory to save plots")
    args = parser.parse_args()
    
    parsed_data = parse_logs(args.logs)
    plot_metrics(parsed_data, args.output)
    print(f"Plots saved to {args.output}")
