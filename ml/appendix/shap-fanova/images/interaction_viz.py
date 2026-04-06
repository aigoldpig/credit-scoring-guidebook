"""
fANOVA 교호작용 시각화
- Figure 1: 2-way interaction — depth-2 (3x3, 깔끔) vs 깊은 트리 누적 split (7x6)
- Figure 2: 3-way interaction cube
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ════════════════════════════════════════════════════════════════
# Figure 1: 2-way interaction — 깔끔한 3×3 vs 깊은 트리 누적 split
# ════════════════════════════════════════════════════════════════

fig1, (ax_simple, ax_real) = plt.subplots(1, 2, figsize=(16, 7.2),
                                           gridspec_kw={'width_ratios': [1, 1.8],
                                                        'wspace': 0.35})

cmap = plt.cm.RdBu_r
norm = TwoSlopeNorm(vmin=-0.30, vcenter=0, vmax=0.30)

# ── (a) depth-2: split 2개씩 → 3×3 = 9 셀 ──
interaction_simple = np.array([
    [+0.12,  +0.05,  -0.18],   # DTI 高
    [+0.03,   0.00,  -0.04],   # DTI 中
    [-0.15,  -0.06,  +0.22],   # DTI 低
])
n_row_s, n_col_s = interaction_simple.shape

for i in range(n_row_s):
    for j in range(n_col_s):
        val = interaction_simple[i, j]
        color = cmap(norm(val))
        rect = patches.FancyBboxPatch(
            (j, n_row_s - 1 - i), 1, 1,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='#bbbbbb', linewidth=1.2
        )
        ax_simple.add_patch(rect)
        sign = '+' if val > 0 else ''
        ax_simple.text(j + 0.5, n_row_s - 1 - i + 0.5, f'{sign}{val:.2f}',
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       color='white' if abs(val) > 0.10 else '#333333')

# 분기점 라인
for x in range(1, n_col_s):
    ax_simple.axvline(x, color='#444444', linewidth=1.5, linestyle='--', alpha=0.6)
for y in range(1, n_row_s):
    ax_simple.axhline(y, color='#444444', linewidth=1.5, linestyle='--', alpha=0.6)

ax_simple.set_xlim(-0.05, n_col_s + 0.05)
ax_simple.set_ylim(-0.15, n_row_s + 0.35)
ax_simple.set_xticks([0.5, 1.5, 2.5])
ax_simple.set_xticklabels(['< 3,000', '3,000\n~6,000', '> 6,000'], fontsize=9)
ax_simple.set_yticks([0.5, 1.5, 2.5])
ax_simple.set_yticklabels(['< 40%', '40~80%', '> 80%'], fontsize=9)
ax_simple.set_xlabel('$X_1$: 소득 (만 원)', fontsize=11, labelpad=6)
ax_simple.set_ylabel('$X_2$: DTI (%)', fontsize=11, labelpad=6)
ax_simple.set_title('(a)  depth-2 트리\n$X_1$ split 2개 $\\times$ $X_2$ split 2개 = 9 셀',
                     fontsize=12, fontweight='bold', pad=10)
ax_simple.set_aspect('equal')
for spine in ax_simple.spines.values():
    spine.set_visible(False)

# ── (b) 깊은 트리: 누적 split 6개 × 5개 → 7×6 = 42 셀 ──
# 현실적인 비단조 패턴: 인접 셀끼리 부호가 뒤집히는 구간 다수
np.random.seed(42)

n_row_r, n_col_r = 6, 7  # DTI 6구간, 소득 7구간

# 기본 패턴 생성 + 비단조성 주입
base = np.outer(np.linspace(0.20, -0.25, n_row_r),
                np.linspace(-0.15, 0.20, n_col_r))
noise = np.random.uniform(-0.12, 0.12, (n_row_r, n_col_r))
interaction_real = base + noise
# 부호 뒤집기: 일부 셀에서 의도적 비단조 (인접 셀과 부호 반전)
interaction_real[1, 3] = +0.16   # 주변은 음수인데 갑자기 양수
interaction_real[3, 1] = -0.20   # 주변은 양수인데 갑자기 음수
interaction_real[4, 5] = +0.15   # 주변은 음수인데 갑자기 양수
interaction_real[0, 2] = -0.13   # 첫 행에서도 부호 반전
interaction_real = np.clip(interaction_real, -0.30, 0.30)

# 소득 분기점 라벨 (7구간 → 6개 split)
x1_splits_real = ['1.5k', '2.5k', '3.5k', '4.5k', '6k', '8k']
x2_splits_real = ['20%', '35%', '50%', '70%', '90%']

for i in range(n_row_r):
    for j in range(n_col_r):
        val = interaction_real[i, j]
        color = cmap(norm(val))
        rect = patches.Rectangle(
            (j, n_row_r - 1 - i), 1, 1,
            facecolor=color, edgecolor='#cccccc', linewidth=0.8
        )
        ax_real.add_patch(rect)

        # 값 표시 (작은 폰트, 빽빽)
        sign = '+' if val > 0 else ''
        ax_real.text(j + 0.5, n_row_r - 1 - i + 0.5, f'{sign}{val:.2f}',
                     ha='center', va='center', fontsize=7.5, fontweight='bold',
                     color='white' if abs(val) > 0.12 else '#444444')

# 분기점 라인
for x in range(1, n_col_r):
    ax_real.axvline(x, color='#444444', linewidth=1, linestyle='--', alpha=0.4)
for y in range(1, n_row_r):
    ax_real.axhline(y, color='#444444', linewidth=1, linestyle='--', alpha=0.4)

# 비단조 구간 강조 — 인접 셀과 부호가 반전되는 셀
nonmono_cells = [(1, 3), (3, 1), (4, 5), (0, 2)]

for (r, c) in nonmono_cells:
    cx, cy = c + 0.5, n_row_r - 1 - r + 0.5
    circle = plt.Circle((cx, cy), 0.42, fill=False, edgecolor='#ff4444',
                         linewidth=2.5, linestyle='-')
    ax_real.add_patch(circle)

ax_real.set_xlim(-0.05, n_col_r + 0.05)
ax_real.set_ylim(-0.4, n_row_r + 0.45)
# x축: 구간 라벨
x_mid = [j + 0.5 for j in range(n_col_r)]
x_labels_r = ['<1.5k', '~2.5k', '~3.5k', '~4.5k', '~6k', '~8k', '>8k']
ax_real.set_xticks(x_mid)
ax_real.set_xticklabels(x_labels_r, fontsize=7.5)
y_mid = [n_row_r - 1 - i + 0.5 for i in range(n_row_r)]
y_labels_r = ['>90%', '70~90', '50~70', '35~50', '20~35', '<20%']
ax_real.set_yticks(y_mid)
ax_real.set_yticklabels(y_labels_r, fontsize=7.5)
ax_real.set_xlabel('$X_1$: 소득 (만 원)', fontsize=11, labelpad=6)
ax_real.set_ylabel('$X_2$: DTI (%)', fontsize=11, labelpad=6)
ax_real.set_title('(b)  깊은 트리에서 $X_1$, $X_2$의 누적 split\n$X_1$ 6개 $\\times$ $X_2$ 5개 = 42 셀',
                  fontsize=12, fontweight='bold', pad=10)

# 비단조 범례
ax_real.plot([], [], 'o', mfc='none', mec='#ff4444', mew=2.5, ms=12,
             label='비단조 셀: 인접 셀과 효과 방향이 반전')
ax_real.legend(loc='lower left', fontsize=9, framealpha=0.9)

ax_real.set_aspect('equal')
for spine in ax_real.spines.values():
    spine.set_visible(False)

# 공유 컬러바
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig1.add_axes([0.84, 0.15, 0.02, 0.65])
cbar = fig1.colorbar(sm, cax=cbar_ax)
cbar.set_label('교호작용 효과 $f_{12}$ (logit)\n빨강: 불량 확률 ↑  /  파랑: 정상 확률 ↑', fontsize=10)

fig1.suptitle('$f_{12}(X_1, X_2)$:  트리가 깊어지면 교호작용 해석이 어려워진다',
              fontsize=14, fontweight='bold', y=1.02)
fig1.subplots_adjust(right=0.82, left=0.04)
fig1.savefig('interaction_2way.png', dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
print("OK: interaction_2way.png saved")


# ════════════════════════════════════════════════════════════════
# Figure 2: 3-way interaction cube
# ════════════════════════════════════════════════════════════════

interaction_3d = np.array([
    [[-0.08, +0.12], [+0.15, -0.20]],   # 연체 경과 < 3개월
    [[+0.18, -0.10], [-0.05, +0.06]],   # 연체 경과 >= 3개월
])

def draw_cube(ax, x0, y0, z0, dx, dy, dz, color, alpha=0.6, edgecolor='#555555'):
    vertices = np.array([
        [x0, y0, z0], [x0+dx, y0, z0], [x0+dx, y0+dy, z0], [x0, y0+dy, z0],
        [x0, y0, z0+dz], [x0+dx, y0, z0+dz], [x0+dx, y0+dy, z0+dz], [x0, y0+dy, z0+dz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
    ]
    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                   edgecolor=edgecolor, linewidth=0.8)
    ax.add_collection3d(collection)

fig2 = plt.figure(figsize=(9, 7.5))
ax2 = fig2.add_subplot(111, projection='3d')

cmap3 = plt.cm.RdBu_r
norm3 = TwoSlopeNorm(vmin=-0.22, vcenter=0, vmax=0.22)

gap = 0.08
for iz in range(2):
    for iy in range(2):
        for ix in range(2):
            val = interaction_3d[iz, iy, ix]
            color = cmap3(norm3(val))
            x0 = ix * (1 + gap)
            y0 = iy * (1 + gap)
            z0 = iz * (1 + gap)
            alpha = 0.35 + 0.35 * (abs(val) / 0.20)
            draw_cube(ax2, x0, y0, z0, 1, 1, 1, color, alpha=min(alpha, 0.85))
            sign = '+' if val > 0 else ''
            ax2.text(x0 + 0.5, y0 + 0.5, z0 + 0.5,
                     f'{sign}{val:.2f}',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='#222222')

ax2.set_xlabel('\n$X_1$: 소득', fontsize=11, labelpad=12)
ax2.set_ylabel('\n$X_2$: DTI', fontsize=11, labelpad=12)
ax2.set_zlabel('\n$X_3$: 최근 연체 경과기간', fontsize=11, labelpad=12)
ax2.set_xticks([0.5, 1.5 + gap])
ax2.set_xticklabels(['< 4,000만', '>= 4,000만'], fontsize=8)
ax2.set_yticks([0.5, 1.5 + gap])
ax2.set_yticklabels(['DTI >= 60%', 'DTI < 60%'], fontsize=8)
ax2.set_zticks([0.5, 1.5 + gap])
ax2.set_zticklabels(['< 3개월', '>= 3개월'], fontsize=8)
ax2.set_title('$f_{123}(X_1, X_2, X_3)$: 3-way 교호작용\n8개 셀의 방향과 크기를 직관적으로 파악할 수 있는가?\n빨강: 불량 확률 ↑  /  파랑: 정상 확률 ↑',
              fontsize=12, fontweight='bold', pad=20)
ax2.view_init(elev=22, azim=-55)
ax2.set_xlim(-0.1, 2.2)
ax2.set_ylim(-0.1, 2.2)
ax2.set_zlim(-0.1, 2.2)

sm3 = plt.cm.ScalarMappable(cmap=cmap3, norm=norm3)
sm3.set_array([])
cbar3 = fig2.colorbar(sm3, ax=ax2, shrink=0.55, pad=0.08)
cbar3.set_label('교호작용 효과 (logit)\n빨강: 불량↑ / 파랑: 정상↑', fontsize=10)

fig2.tight_layout()
fig2.savefig('interaction_3way_cube.png', dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
print("OK: interaction_3way_cube.png saved")
