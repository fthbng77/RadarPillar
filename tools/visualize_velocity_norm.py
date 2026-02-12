"""
Velocity normalizasyonunun etkisini görselleştir.
Gerçek VoD radar verisinden vx/vy dağılımlarını çizer.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path('./data/VoD/view_of_delft_PUBLIC/radar_5frames')

# Config'teki normalizasyon parametreleri
MEAN_VX, MEAN_VY = 0.020094, -0.003165
STD_VX, STD_VY = 0.891514, 0.452536

# VoD radar feature sırası: [x, y, z, rcs, v_r, v_r_comp, time]
V_COMP_INDEX = 5  # v_r_comp


def load_all_points():
    """Train setindeki tüm noktaları yükle."""
    info_path = DATA_ROOT / 'vod_infos_train.pkl'
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    all_vx, all_vy = [], []
    for info in infos:
        lidar_idx = info['point_cloud']['lidar_idx']
        bin_path = DATA_ROOT / 'training' / 'velodyne' / f'{lidar_idx}.bin'

        if not bin_path.exists():
            continue

        points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 7)
        x, y = points[:, 0], points[:, 1]
        v_comp = points[:, V_COMP_INDEX]

        # Azimut açısı ile kartezyen ayrıştırma
        phi = np.arctan2(y, x)
        vx = v_comp * np.cos(phi)
        vy = v_comp * np.sin(phi)

        all_vx.append(vx)
        all_vy.append(vy)

    return np.concatenate(all_vx), np.concatenate(all_vy)


def main():
    print("VoD train verisi yükleniyor...")
    vx_raw, vy_raw = load_all_points()
    print(f"Toplam {len(vx_raw):,} nokta yüklendi")
    print(f"vx: mean={vx_raw.mean():.4f}, std={vx_raw.std():.4f}, min={vx_raw.min():.2f}, max={vx_raw.max():.2f}")
    print(f"vy: mean={vy_raw.mean():.4f}, std={vy_raw.std():.4f}, min={vy_raw.min():.2f}, max={vy_raw.max():.2f}")

    # Normalize
    vx_norm = (vx_raw - MEAN_VX) / STD_VX
    vy_norm = (vy_raw - MEAN_VY) / STD_VY

    # --- Figür ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Velocity Normalizasyonu Etkisi (VoD Radar Train Set)', fontsize=16, fontweight='bold')

    # Row 1: Ham (normalizasyonsuz)
    axes[0, 0].hist(vx_raw, bins=200, range=(-5, 5), color='#e74c3c', alpha=0.7, density=True)
    axes[0, 0].set_title('vx (Ham)', fontsize=13)
    axes[0, 0].set_xlabel('m/s')
    axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_ylabel('Yoğunluk')
    stats_text = f'μ={vx_raw.mean():.3f}\nσ={vx_raw.std():.3f}'
    axes[0, 0].text(0.95, 0.95, stats_text, transform=axes[0, 0].transAxes,
                    va='top', ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0, 1].hist(vy_raw, bins=200, range=(-5, 5), color='#3498db', alpha=0.7, density=True)
    axes[0, 1].set_title('vy (Ham)', fontsize=13)
    axes[0, 1].set_xlabel('m/s')
    axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    stats_text = f'μ={vy_raw.mean():.3f}\nσ={vy_raw.std():.3f}'
    axes[0, 1].text(0.95, 0.95, stats_text, transform=axes[0, 1].transAxes,
                    va='top', ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2D scatter (ham)
    subsample = np.random.choice(len(vx_raw), min(50000, len(vx_raw)), replace=False)
    axes[0, 2].scatter(vx_raw[subsample], vy_raw[subsample], s=0.3, alpha=0.15, c='#e74c3c')
    axes[0, 2].set_title('vx vs vy (Ham)', fontsize=13)
    axes[0, 2].set_xlabel('vx (m/s)')
    axes[0, 2].set_ylabel('vy (m/s)')
    axes[0, 2].set_xlim(-5, 5)
    axes[0, 2].set_ylim(-5, 5)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].axvline(0, color='k', linestyle='--', alpha=0.3)

    # Row 2: Normalize edilmiş
    axes[1, 0].hist(vx_norm, bins=200, range=(-5, 5), color='#2ecc71', alpha=0.7, density=True)
    axes[1, 0].set_title('vx (Normalize)', fontsize=13)
    axes[1, 0].set_xlabel('z-score')
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_ylabel('Yoğunluk')
    stats_text = f'μ={vx_norm.mean():.3f}\nσ={vx_norm.std():.3f}'
    axes[1, 0].text(0.95, 0.95, stats_text, transform=axes[1, 0].transAxes,
                    va='top', ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1, 1].hist(vy_norm, bins=200, range=(-5, 5), color='#9b59b6', alpha=0.7, density=True)
    axes[1, 1].set_title('vy (Normalize)', fontsize=13)
    axes[1, 1].set_xlabel('z-score')
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    stats_text = f'μ={vy_norm.mean():.3f}\nσ={vy_norm.std():.3f}'
    axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    va='top', ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2D scatter (normalize)
    axes[1, 2].scatter(vx_norm[subsample], vy_norm[subsample], s=0.3, alpha=0.15, c='#2ecc71')
    axes[1, 2].set_title('vx vs vy (Normalize)', fontsize=13)
    axes[1, 2].set_xlabel('vx (z-score)')
    axes[1, 2].set_ylabel('vy (z-score)')
    axes[1, 2].set_xlim(-5, 5)
    axes[1, 2].set_ylim(-5, 5)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_path = 'tools/velocity_normalization_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Kaydedildi: {out_path}")
    plt.close()

    # --- Ek: Feature ölçek karşılaştırması ---
    fig2, ax = plt.subplots(figsize=(10, 5))
    feature_names = ['x', 'y', 'z', 'rcs', 'vx\n(ham)', 'vy\n(ham)', 'vx\n(norm)', 'vy\n(norm)']
    # x,y,z,rcs istatistiklerini de hesaplayalım
    # Tüm veriden ilk 200 frame yeterli
    info_path = DATA_ROOT / 'vod_infos_train.pkl'
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    sample_points = []
    for info in infos[:200]:
        lidar_idx = info['point_cloud']['lidar_idx'] if isinstance(info['point_cloud'], dict) else info['point_cloud']
        bin_path = DATA_ROOT / 'training' / 'velodyne' / f'{lidar_idx}.bin'
        if bin_path.exists():
            pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 7)
            sample_points.append(pts)
    sample_pts = np.concatenate(sample_points)

    stds = [
        sample_pts[:, 0].std(),  # x
        sample_pts[:, 1].std(),  # y
        sample_pts[:, 2].std(),  # z
        sample_pts[:, 3].std(),  # rcs
        vx_raw.std(),            # vx ham
        vy_raw.std(),            # vy ham
        vx_norm.std(),           # vx norm
        vy_norm.std(),           # vy norm
    ]

    colors = ['#95a5a6'] * 4 + ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
    bars = ax.bar(feature_names, stds, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Standart Sapma (σ)', fontsize=12)
    ax.set_title('Feature Ölçekleri: Normalizasyon Öncesi vs Sonrası', fontsize=14, fontweight='bold')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.4, label='σ = 1.0 (ideal)')
    ax.legend(fontsize=11)

    for bar, val in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path2 = 'tools/feature_scale_comparison.png'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Kaydedildi: {out_path2}")
    plt.close()


if __name__ == '__main__':
    main()
