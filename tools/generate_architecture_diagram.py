"""
Generate RadarPillar architecture diagram for README.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_block(ax, x, y, w, h, title, details, color, text_color='white',
               fontsize_title=11, fontsize_detail=8):
    """Draw a rounded rectangle block with title and details."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15", linewidth=1.5,
                          edgecolor='#333333', facecolor=color, zorder=2)
    ax.add_patch(box)

    if details:
        ax.text(x, y + h*0.15, title, ha='center', va='center',
                fontsize=fontsize_title, fontweight='bold', color=text_color, zorder=3)
        ax.text(x, y - h*0.2, details, ha='center', va='center',
                fontsize=fontsize_detail, color=text_color, alpha=0.9, zorder=3,
                style='italic')
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=fontsize_title, fontweight='bold', color=text_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                connectionstyle='arc3,rad=0'))


def draw_side_label(ax, x, y, text, color='#666666'):
    """Draw a side annotation label."""
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
            color=color, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                      edgecolor='#cccccc', alpha=0.9))


def generate_diagram(out_path):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # Color palette
    C_INPUT   = '#2C3E50'
    C_VFE     = '#2980B9'
    C_ATTN    = '#8E44AD'
    C_SCATTER = '#27AE60'
    C_BEV     = '#E67E22'
    C_HEAD    = '#C0392B'
    C_OUTPUT  = '#2C3E50'
    C_DECOMP  = '#3498DB'

    bw, bh = 5.0, 1.2   # block width, height
    cx = 7.0             # center x

    # ── Blocks (top to bottom) ──
    # Input
    y_input = 12.0
    draw_block(ax, cx, y_input, bw, bh,
               'Radar Point Cloud', '(N, 7): x, y, z, RCS, v_r, v_r_comp, time',
               C_INPUT, fontsize_title=12)

    # PillarVFE
    y_vfe = 10.0
    draw_block(ax, cx, y_vfe, bw+1, bh+0.3,
               'PillarVFE', 'Voxelization  +  Velocity Decomposition',
               C_VFE, fontsize_title=12)

    # Velocity decomposition detail box (side)
    draw_side_label(ax, cx + 5.2, y_vfe + 0.25,
                    r'$\phi$ = atan2(y, x)')
    draw_side_label(ax, cx + 5.2, y_vfe - 0.25,
                    r'$v_x$ = $v_r \cdot$ cos($\phi$),  $v_y$ = $v_r \cdot$ sin($\phi$)')
    # connect side labels
    ax.annotate('', xy=(cx + 3.5, y_vfe), xytext=(cx + 3.85, y_vfe),
                arrowprops=dict(arrowstyle='-', color='#999999', lw=1, ls='--'))

    # Arrow input -> VFE
    draw_arrow(ax, cx, y_input - bh/2, cx, y_vfe + (bh+0.3)/2)

    # Pillar features label
    draw_side_label(ax, cx - 4.5, (y_vfe + 8.0)/2,
                    'Pillar features (M, 32)')

    # PillarAttention
    y_attn = 7.8
    draw_block(ax, cx, y_attn, bw+1, bh+0.3,
               'PillarAttention', 'Masked Multi-Head Self-Attention (C=32, H=1)',
               C_ATTN, fontsize_title=12)

    draw_side_label(ax, cx + 5.5, y_attn,
                    'LayerNorm + FFN + Residual')

    draw_arrow(ax, cx, y_vfe - (bh+0.3)/2, cx, y_attn + (bh+0.3)/2)

    # PointPillarScatter
    y_scatter = 5.8
    draw_block(ax, cx, y_scatter, bw, bh,
               'PointPillarScatter', 'Sparse  →  Dense BEV Grid (320×320×32)',
               C_SCATTER, fontsize_title=11)

    draw_arrow(ax, cx, y_attn - (bh+0.3)/2, cx, y_scatter + bh/2)

    # BaseBEVBackbone
    y_bev = 3.8
    draw_block(ax, cx, y_bev, bw+1, bh+0.3,
               'BaseBEVBackbone', '3-layer 2D CNN (32ch) + Multi-scale Upsample',
               C_BEV, fontsize_title=12)

    draw_side_label(ax, cx - 5.0, y_bev,
                    'Strides: [2, 2, 2]\nFilters: [32, 32, 32]')

    draw_arrow(ax, cx, y_scatter - bh/2, cx, y_bev + (bh+0.3)/2)

    # AnchorHeadSingle
    y_head = 1.8
    draw_block(ax, cx, y_head, bw+1, bh+0.3,
               'AnchorHeadSingle', '3 Classes + Direction Classifier + NMS',
               C_HEAD, fontsize_title=12)

    draw_side_label(ax, cx + 5.5, y_head,
                    'Car | Pedestrian | Cyclist')

    draw_arrow(ax, cx, y_bev - (bh+0.3)/2, cx, y_head + (bh+0.3)/2)

    # Output
    y_out = 0.0
    draw_block(ax, cx, y_out, bw, bh,
               '3D Bounding Boxes', '(x, y, z, dx, dy, dz, heading, score)',
               C_OUTPUT, fontsize_title=12)

    draw_arrow(ax, cx, y_head - (bh+0.3)/2, cx, y_out + bh/2)

    # ── Title ──
    fig.suptitle('RadarPillar Architecture — VoD Configuration',
                 fontsize=16, fontweight='bold', y=0.98, color='#2C3E50')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    generate_diagram('docs/visualizations/radarpillar_architecture.png')
