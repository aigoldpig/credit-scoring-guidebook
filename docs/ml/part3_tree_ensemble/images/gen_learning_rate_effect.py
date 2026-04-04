"""Learning Rate에 따른 GBM 수렴 패턴 — 올바른 시뮬레이션.

Train loss는 라운드가 늘수록 단조 감소해야 한다.
Valid loss는 과적합 시점 이후 상승(U자)한다.
η가 작을수록 과적합 시점이 늦고, 최종 valid loss 최솟값이 낮다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 ──
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


def make_curves(n_rounds, eta_label, train_floor, train_speed,
                valid_min, valid_min_round, valid_rise_speed, noise_std):
    """Train/Valid loss 곡선 생성.

    - Train: 지수 감소 (단조 하강) + 작은 노이즈
    - Valid: 지수 감소 후 서서히 상승 (U자) + 노이즈
    """
    t = np.arange(1, n_rounds + 1)

    # Train loss: 단조 감소 — 지수 감소 곡선
    train_base = train_floor + (0.50 - train_floor) * np.exp(-train_speed * t)
    # 단조성 보장: cumulative minimum
    train_base = np.minimum.accumulate(train_base)

    # Valid loss: U자 — 감소 후 상승
    decay = (0.50 - valid_min) * np.exp(-train_speed * 0.8 * t)
    rise = valid_rise_speed * np.maximum(t - valid_min_round, 0) ** 1.3
    valid_base = valid_min + decay + rise

    # 노이즈 추가
    train_noisy = train_base + np.random.normal(0, noise_std * 0.3, n_rounds)
    valid_noisy = valid_base + np.random.normal(0, noise_std, n_rounds)

    # Train은 단조 감소 보장 (노이즈 후에도)
    train_noisy = np.minimum.accumulate(train_noisy)
    # 초기값 고정
    train_noisy[0] = 0.50
    valid_noisy[0] = 0.50

    return t, train_noisy, valid_noisy, valid_min_round


# ── 세 시나리오 파라미터 ──
configs = [
    {
        'n_rounds': 200,
        'eta_label': r'$\eta$ = 0.3 (큰 학습률)',
        'color': '#e74c3c',
        'train_floor': 0.22,
        'train_speed': 0.025,
        'valid_min': 0.265,
        'valid_min_round': 7,
        'valid_rise_speed': 0.0008,
        'noise_std': 0.008,
    },
    {
        'n_rounds': 600,
        'eta_label': r'$\eta$ = 0.1 (중간)',
        'color': '#e67e22',
        'train_floor': 0.18,
        'train_speed': 0.008,
        'valid_min': 0.250,
        'valid_min_round': 45,
        'valid_rise_speed': 0.00012,
        'noise_std': 0.008,
    },
    {
        'n_rounds': 2000,
        'eta_label': r'$\eta$ = 0.01 (작은 학습률)',
        'color': '#2980b9',
        'train_floor': 0.15,
        'train_speed': 0.002,
        'valid_min': 0.238,
        'valid_min_round': 200,
        'valid_rise_speed': 0.0000012,
        'noise_std': 0.006,
    },
]

# ── 그리기 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
fig.suptitle('Learning Rate에 따른 GBM 수렴 패턴', fontsize=16, fontweight='bold', y=1.02)

for ax, cfg in zip(axes, configs):
    t, train_loss, valid_loss, best_round = make_curves(
        cfg['n_rounds'], cfg['eta_label'],
        cfg['train_floor'], cfg['train_speed'],
        cfg['valid_min'], cfg['valid_min_round'],
        cfg['valid_rise_speed'], cfg['noise_std'],
    )

    # 실제 valid 최솟값 라운드 찾기
    actual_best = np.argmin(valid_loss) + 1
    actual_best_loss = valid_loss[actual_best - 1]

    ax.plot(t, train_loss, color=cfg['color'], linewidth=1.2, label='Train Loss')
    ax.plot(t, valid_loss, color='#555555', linewidth=1.0, linestyle='--',
            alpha=0.8, label='Valid Loss')

    # Best 포인트 표시
    ax.plot(actual_best, actual_best_loss, 'o', color='#27ae60', markersize=8, zorder=5)
    ax.axvline(actual_best, color='#27ae60', linestyle=':', alpha=0.6)
    ax.annotate(f'Best: {actual_best}라운드\nLoss={actual_best_loss:.3f}',
                xy=(actual_best, actual_best_loss),
                xytext=(0.45, 0.55), textcoords='axes fraction',
                fontsize=9, color='#27ae60', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#27ae60', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2))

    ax.set_title(cfg['eta_label'], fontsize=13, color=cfg['color'], fontweight='bold')
    ax.set_xlabel('Boosting 라운드', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/ml/part3_tree_ensemble/images/learning_rate_effect.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Done: learning_rate_effect.png')
