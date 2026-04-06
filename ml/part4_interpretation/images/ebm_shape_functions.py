"""
EBM 결과물 시각화 — main effect shape functions + pairwise interaction heatmap
신용평가 맥락의 가상 데이터
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ════════════════════════════════════════════════════════════════
# Figure: EBM Shape Functions — Main Effects + Interaction
# ════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 10), facecolor='white')
gs = GridSpec(2, 3, hspace=0.42, wspace=0.35,
             left=0.06, right=0.96, top=0.90, bottom=0.08)

common_style = dict(color='#0064ff', linewidth=2.5)
fill_style = dict(alpha=0.08, color='#0064ff')
zero_style = dict(color='gray', linewidth=0.7, linestyle='--')

# ── (a) DTI shape function ──
ax1 = fig.add_subplot(gs[0, 0])
x_dti = np.linspace(0, 120, 300)
# 비선형: 40% 이하에서 완만, 40~80%에서 급등, 80% 이상 포화
y_dti = -0.8 + 0.6 * np.tanh((x_dti - 50) / 18) + 0.3 * np.tanh((x_dti - 85) / 12)
ax1.plot(x_dti, y_dti, **common_style)
ax1.fill_between(x_dti, y_dti, **fill_style)
ax1.axhline(0, **zero_style)
ax1.set_xlabel('DTI (%)', fontsize=11)
ax1.set_ylabel('$f$(DTI)  (log-odds)', fontsize=10.5)
ax1.set_title('(a)  DTI', fontsize=12.5, fontweight='bold', pad=8)
ax1.set_xlim(0, 120)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.annotate('DTI 40~80%:\n급격한 위험 증가',
            xy=(60, 0.05), fontsize=9, color='#cc0000',
            ha='center', style='italic')

# ── (b) 소득 shape function ──
ax2 = fig.add_subplot(gs[0, 1])
x_inc = np.linspace(1000, 12000, 300)
# 고소득에서 감소하다가 포화
y_inc = 0.7 - 0.5 * np.tanh((x_inc - 3000) / 1500) - 0.3 * np.tanh((x_inc - 7000) / 2000)
ax2.plot(x_inc, y_inc, **common_style)
ax2.fill_between(x_inc, y_inc, **fill_style)
ax2.axhline(0, **zero_style)
ax2.set_xlabel('소득 (만 원)', fontsize=11)
ax2.set_ylabel('$f$(소득)  (log-odds)', fontsize=10.5)
ax2.set_title('(b)  소득', fontsize=12.5, fontweight='bold', pad=8)
ax2.set_xlim(1000, 12000)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.annotate('고소득 구간:\n효과 포화',
            xy=(9500, -0.08), fontsize=9, color='#0064ff',
            ha='center', style='italic')

# ── (c) 최근 연체 건수 shape function (step-like) ──
ax3 = fig.add_subplot(gs[0, 2])
x_del = np.array([0, 0.5, 0.5, 1, 1.5, 1.5, 2, 2.5, 2.5, 3, 3.5, 3.5, 4, 4.5, 4.5, 5])
y_del = np.array([-0.6, -0.6, -0.2, -0.2, -0.2, 0.15, 0.15, 0.15, 0.45, 0.45, 0.45, 0.62, 0.62, 0.62, 0.70, 0.70])
ax3.step(range(6), [-0.6, 0.10, 0.35, 0.52, 0.62, 0.70],
         where='mid', **common_style)
ax3.bar(range(6), [-0.6, 0.10, 0.35, 0.52, 0.62, 0.70],
        width=0.6, alpha=0.12, color='#0064ff', edgecolor='none')
ax3.axhline(0, **zero_style)
ax3.set_xlabel('최근 12개월 연체 건수', fontsize=11)
ax3.set_ylabel('$f$(연체건수)  (log-odds)', fontsize=10.5)
ax3.set_title('(c)  최근 연체 건수', fontsize=12.5, fontweight='bold', pad=8)
ax3.set_xticks(range(6))
ax3.set_xticklabels(['0', '1', '2', '3', '4', '5+'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.annotate('연체 1건만으로도\n방향 전환',
            xy=(1.5, -0.05), fontsize=9, color='#cc0000',
            ha='center', style='italic')

# ── (d) 2-way interaction heatmap: DTI × 소득 ──
ax4 = fig.add_subplot(gs[1, 0:2])

dti_bins = ['< 30%', '30~50%', '50~70%', '70~90%', '> 90%']
inc_bins = ['< 2,000', '2,000\n~4,000', '4,000\n~6,000', '6,000\n~8,000', '> 8,000']

# 교호작용: 고DTI+저소득에서 시너지로 위험 증가
interaction = np.array([
    [-0.03, -0.01,  0.00,  0.01,  0.02],   # DTI < 30%
    [-0.02,  0.00,  0.01,  0.01,  0.03],   # 30~50%
    [ 0.01,  0.02,  0.00, -0.02, -0.03],   # 50~70%
    [ 0.04,  0.05,  0.01, -0.04, -0.06],   # 70~90%
    [ 0.08,  0.10,  0.02, -0.07, -0.12],   # > 90%
])

cmap = plt.cm.RdBu_r
norm = TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)

im = ax4.imshow(interaction, cmap=cmap, norm=norm, aspect='auto')

for i in range(5):
    for j in range(5):
        val = interaction[i, j]
        sign = '+' if val > 0 else ''
        textcolor = 'white' if abs(val) > 0.06 else '#333333'
        ax4.text(j, i, f'{sign}{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=textcolor)

ax4.set_xticks(range(5))
ax4.set_xticklabels(inc_bins, fontsize=9)
ax4.set_yticks(range(5))
ax4.set_yticklabels(dti_bins, fontsize=9)
ax4.set_xlabel('소득 (만 원)', fontsize=11, labelpad=8)
ax4.set_ylabel('DTI (%)', fontsize=11, labelpad=8)
ax4.set_title('(d)  DTI × 소득  2-way interaction', fontsize=12.5, fontweight='bold', pad=8)

cbar = fig.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
cbar.set_label('$f$(DTI, 소득)  (log-odds)', fontsize=10)

# ── (e) 해석 요약 텍스트 ──
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

summary_text = (
    "EBM의 결과물\n"
    "━━━━━━━━━━━━━━━━\n\n"
    "① main effect 곡선\n"
    "   → 변수별 line plot\n"
    "   → 어디서 효과가 꺾이는지\n"
    "       한눈에 파악\n\n"
    "② pairwise interaction\n"
    "   → heatmap 한 장\n"
    "   → 어떤 조합이 시너지를\n"
    "       만드는지 직관적\n\n"
    "③ 전체 예측\n"
    "   → 위 항들의 단순 합산\n"
    "   → 어떤 항도 숨겨지지 않음"
)
ax5.text(0.08, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=11.5, verticalalignment='top', fontfamily='Malgun Gothic',
         linespacing=1.5,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4ff',
                   edgecolor='#0064ff', alpha=0.3))

fig.suptitle('EBM이 산출하는 Shape Function — 모형의 모든 판단 근거가 그림으로 드러난다',
             fontsize=14.5, fontweight='bold', y=0.96)

fig.savefig('ebm_shape_functions.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("OK: ebm_shape_functions.png saved")
