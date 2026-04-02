# TabNet

> Tabular 데이터에 특화된 Attention 기반 신경망 아키텍처.

!!! quote "이 장의 핵심"
    트리 앙상블의 장점 — **피처 선택**, **해석 가능성**, **Sparse한 의사결정** — 을 신경망 안에 녹여낸 것이 TabNet이다. "딥러닝으로 Tabular를 풀겠다"는 시도 중에서, 트리의 작동 원리를 가장 직접적으로 모방한 아키텍처다.

!!! quote "저자의 말"
    솔직히 고백하면, TabNet은 이름만 들어봤지 내부 구조를 제대로 들여다본 적이 없었다. 이번 가이드북을 쓰면서 처음으로 논문을 읽어보았고, Sequential Attention, Sparsemax, Self-supervised Pre-training 등 세부 메커니즘은 나에게도 새로운 개념이었다. 이 장의 세부 내용은 **Claude Code가 논문을 기반으로 작성**하였으며, 나는 전체 흐름과 신용평가 맥락에서의 의미를 검토하는 역할을 했다. 틀린 부분이 있을 수 있으니 감안하여 읽어주시기 바란다.

---

## 3.1 TabNet이란

### 배경: 왜 Tabular 전용 신경망이 필요한가

[앞 장](lr_as_nn.md)에서 확인했듯이, Tabular 데이터에서는 MLP가 트리 앙상블에 일관되게 밀린다. 그 핵심 이유 중 하나는 **피처 선택(Feature Selection)** 방식의 차이다.

- **트리**: 각 split에서 **하나의 피처**만 선택하여 분기 → 자연스러운 Sparse 결정
- **MLP**: 모든 뉴런이 **전체 입력**의 가중합을 계산 → Dense 처리, 노이즈 피처에 취약

TabNet(Arik & Pfister, 2021)[^1]은 이 관찰에서 출발한다. 신경망이 **각 의사결정 단계(Step)마다 소수의 피처만 골라서** 사용하도록 강제하면, 트리처럼 Sparse하면서도 신경망의 유연성을 유지할 수 있지 않을까?

!!! info "Google Research"
    TabNet은 Google Research에서 제안되었다. Sercan Arik과 Tomas Pfister가 2019년 arXiv에 초판을 공개하고, 2021년 AAAI에 정식 게재되었다. 논문 제목 자체가 의도를 드러낸다: **"TabNet: Attentive Interpretable Tabular Learning"** — Attention으로 해석 가능한 Tabular 학습.

---

## 3.2 핵심 구조

TabNet의 아키텍처는 **Sequential Attention** 메커니즘을 중심으로 설계되었다. 핵심 아이디어는 의사결정을 **여러 단계(Step)**로 나누고, 각 단계마다 **다른 피처 부분집합**을 선택하는 것이다.

### Step-wise 의사결정

TabNet은 \(N_{\text{steps}}\)개의 단계를 순차적으로 수행한다. 각 Step \(i\)에서:

1. **Attention Transformer**: 이번 Step에서 어떤 피처를 사용할지 결정 (Sparse mask 생성)
2. **Feature Transformer**: 선택된 피처로부터 정보를 추출
3. **집계**: 모든 Step의 출력을 합산하여 최종 예측

$$
\hat{y} = \sum_{i=1}^{N_{\text{steps}}} \text{ReLU}\!\left(h_i(\mathbf{M}_i \odot f(\mathbf{x}))\right)
\tag{1}
$$

여기서 \(\mathbf{M}_i\)는 Step \(i\)의 **Attention mask** (어떤 피처를 볼지), \(f(\mathbf{x})\)는 입력의 초기 변환, \(h_i\)는 Feature Transformer, \(\odot\)는 원소별 곱(element-wise product)이다.

### Sparse Attention Mask

TabNet의 핵심은 **Sparsemax** 활성함수를 사용한 Attention mask다.

일반적인 Softmax는 모든 피처에 0보다 큰 가중치를 부여하지만, Sparsemax는 대부분의 가중치를 **정확히 0**으로 만든다. 이것이 트리의 "하나의 피처만 선택"하는 성질을 신경망에서 구현하는 핵심 장치다.

$$
\mathbf{M}_i = \text{Sparsemax}\!\left(\mathbf{P}_{i-1} \cdot h_{\text{attn}}\!\left(a_{i-1}\right)\right)
\tag{2}
$$

\(\mathbf{P}_{i-1}\)은 **Prior scales** — 이전 Step에서 이미 많이 사용한 피처의 가중치를 낮추는 역할을 한다. 즉, 각 Step이 **서로 다른 피처에 집중**하도록 유도한다.

!!! tip "트리 앙상블과의 비유"
    | 트리 앙상블 | TabNet |
    |---|---|
    | 각 노드에서 **1개 피처**로 split | 각 Step에서 **소수의 피처**를 Sparse mask로 선택 |
    | 여러 트리가 서로 다른 피처 조합 사용 | 여러 Step이 서로 다른 피처 조합 사용 |
    | Feature Importance = split 빈도 | Feature Importance = Attention mask 합산 |
    | Greedy하게 최적 split 탐색 | Gradient descent로 최적 mask 학습 |

### Feature Transformer

선택된 피처는 **Feature Transformer** 블록을 통과한다. 이 블록은 Shared layers와 Step-specific layers로 구성되며, BatchNorm과 GLU(Gated Linear Unit)를 사용한다.

$$
\text{GLU}(\mathbf{x}) = \mathbf{x}_1 \odot \sigma(\mathbf{x}_2)
\tag{3}
$$

여기서 입력을 두 부분으로 나누어, 한쪽은 정보를 담고(\(\mathbf{x}_1\)), 다른 쪽은 게이트 역할(\(\sigma(\mathbf{x}_2)\))을 한다. LSTM의 게이트와 유사한 메커니즘이다.

!!! note "Shared vs. Step-specific"
    Feature Transformer의 일부 층은 모든 Step이 **공유**하고, 나머지는 각 Step이 **독립적으로** 보유한다. 공유 층은 모든 피처 조합에 공통적인 패턴을 학습하고, 독립 층은 각 Step의 고유한 표현을 학습한다. 이 설계가 파라미터 효율성과 표현력 사이의 균형을 맞춘다.

### Sparsity Regularization

TabNet은 Attention mask의 Sparsity를 명시적으로 정규화한다.

$$
L_{\text{sparse}} = \sum_{i=1}^{N_{\text{steps}}} \sum_{j=1}^{D} \frac{-M_{i,j} \log(M_{i,j} + \epsilon)}{N_{\text{steps}}}
\tag{4}
$$

이 항은 **엔트로피 정규화**로, 각 Step의 mask가 소수의 피처에 집중하도록 유도한다. 하이퍼파라미터 \(\lambda_{\text{sparse}}\)로 Sparsity 강도를 조절한다.

---

## 3.3 해석 가능성: Instance-wise Feature Importance

TabNet이 "Interpretable"을 제목에 내건 이유는 **Attention mask에서 직접 Feature Importance를 추출**할 수 있기 때문이다.

### Global Feature Importance

모든 샘플에 대해 Attention mask를 합산하면, 어떤 피처가 전체적으로 중요한지 파악된다.

$$
\text{Importance}_j = \sum_{i=1}^{N_{\text{steps}}} \frac{1}{N} \sum_{n=1}^{N} \eta_i \cdot M_{i,j}^{(n)}
\tag{5}
$$

여기서 \(\eta_i\)는 각 Step의 집계 계수다.

### Instance-wise Importance

샘플별로 Attention mask를 확인하면, **이 고객의 예측에 어떤 변수가 기여했는지** 개별적으로 볼 수 있다. 이 점이 SHAP 같은 사후(post-hoc) 해석 기법과 구별되는 **내재적(intrinsic) 해석 가능성**이다.

!!! warning "해석 가능성의 한계"
    TabNet의 Attention mask는 "이 Step에서 어떤 피처를 **봤는지**"를 알려줄 뿐, 그 피처가 예측에 **어떤 방향으로** 기여했는지(양 or 음)는 직접 보여주지 않는다. Feature Transformer 내부의 비선형 변환을 거치기 때문이다. 완전한 해석을 위해서는 여전히 SHAP 등 사후 분석이 필요할 수 있다.

    트리 앙상블의 split-based Feature Importance도 같은 한계를 공유한다 — 중요도는 알 수 있지만 방향(+/-)은 별도 분석이 필요하다. 이 점에서 TabNet의 해석 가능성이 트리보다 "더 좋다"고 단정하기는 어렵다.

---

## 3.4 Self-supervised Pre-training

TabNet의 또 다른 특징은 **Self-supervised Pre-training**을 지원한다는 점이다.

라벨이 없는 대량의 데이터로 먼저 TabNet의 인코더를 학습시킨 뒤, 라벨이 있는 데이터로 Fine-tuning한다. 학습 방법은 **Masked Feature Prediction** — 입력 피처의 일부를 마스킹하고, 나머지 피처로 마스킹된 값을 복원하는 것이다.

!!! example "신용평가에서의 의미"
    신용평가 데이터는 라벨(부도 여부) 확정까지 **성과 관찰 기간**(보통 12~24개월)이 필요하다. 라벨이 아직 확정되지 않은 최근 신청 데이터는 풍부하지만, 라벨이 있는 데이터는 제한적이다.

    Self-supervised Pre-training은 라벨 없는 데이터로 피처 간 관계를 먼저 학습하고, 소량의 라벨 데이터로 Fine-tuning하는 전략을 가능하게 한다. 논문에서는 이 방식이 **라벨이 적은 상황에서 성능 향상에 기여**한다고 보고한다.

---

## 3.5 신용평가에서의 위치

### GBDT와의 비교

| | TabNet | GBDT (XGBoost/LightGBM) |
|---|---|---|
| **피처 선택** | Attention mask (학습 기반) | Split 기반 (Greedy) |
| **해석 가능성** | Attention mask → Feature Importance | Split Importance, SHAP |
| **결측치 처리** | 학습으로 처리 가능 | 네이티브 처리 |
| **범주형 변수** | Embedding 필요 | 네이티브 (LightGBM, CatBoost) |
| **하이퍼파라미터** | 다수 (Steps, mask 구조, LR, ...) | 상대적으로 적음 |
| **학습 속도** | 느림 (GPU 필요) | 빠름 (CPU로 충분) |
| **재현성** | GPU 비결정성 이슈 | 시드 고정으로 완벽 재현 |
| **Pre-training** | 지원 (Masked Feature Prediction) | 미지원 |

### 벤치마크 결과

TabNet 원 논문에서는 다수의 Tabular 벤치마크에서 트리 앙상블과 **경쟁적인 성능**을 보고했다. 그러나 이후의 대규모 독립 벤치마크에서는 결과가 다소 엇갈린다.

- **Grinsztajn et al. (2022)**[^2]: 45개 데이터셋에서 트리 기반 모형이 TabNet 포함 DL 모형을 일관되게 앞섬
- **Shwartz-Ziv & Armon (2022)**[^3]: XGBoost가 대부분의 DL 모형과 대등하거나 우세
- **McElfresh et al. (2023)**[^4]: 176개 데이터셋 메타분석에서도 GBDT가 평균적으로 우세

!!! note "공정한 평가"
    TabNet이 트리에 밀리는 이유 중 하나는 **튜닝 난이도**다. GBDT는 기본 하이퍼파라미터로도 합리적인 성능을 내지만, TabNet은 \(N_{\text{steps}}\), \(N_a\), \(N_d\), \(\lambda_{\text{sparse}}\), 학습률, 배치 크기 등 **조정할 파라미터가 많고 민감도가 높다**. Kadra et al.(2021)[^5]이 보여줬듯, 충분히 튜닝된 MLP도 GBDT에 근접하지만 그 튜닝 비용이 비현실적이었다. TabNet도 비슷한 문제를 안고 있다.

### 실무 관점

신용평가 실무에서 TabNet의 입지는 다음과 같이 정리된다.

1. **XGBoost/LightGBM이 기본 선택지** — Tabular 데이터에서 성능, 속도, 재현성, 해석 도구 모든 면에서 검증됨
2. **TabNet은 특정 조건에서 고려** — 라벨이 부족한 상황에서 Self-supervised Pre-training이 유리할 수 있음
3. **해석 가능성은 기대보다 제한적** — Attention mask만으로 규제 요건을 충족하기 어려움. SHAP 등 추가 분석은 여전히 필요

!!! tip "그래서 언제 써야 하나?"
    현실적으로 TabNet이 GBDT를 대체할 명확한 우위 시나리오는 좁다. 하지만 **"딥러닝으로 Tabular를 풀려면 어떤 구조가 필요한가"**를 이해하는 데는 매우 교육적인 아키텍처다.

    - 트리의 Sparse split → Sparsemax Attention
    - 단계적 피처 선택 → Sequential multi-step 구조
    - 앙상블의 다양성 → Step별 다른 피처 조합

    TabNet은 "트리가 왜 Tabular에서 강한가"에 대한 답을 **신경망 언어로 번역한 것**이다. 트리가 이기는 이유를 이해하면, TabNet의 설계 동기도 자연스럽게 이해된다.

---

## 3.6 pytorch-tabnet 실습 참고

TabNet의 대표적인 오픈소스 구현은 `pytorch-tabnet` 라이브러리다.

```python
from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(
    n_d=64, n_a=64,          # Feature Transformer 차원
    n_steps=5,                # Attention step 수
    gamma=1.5,                # Prior scales 감쇠 계수
    lambda_sparse=1e-4,       # Sparsity 정규화 강도
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="sparsemax",    # 또는 "entmax"
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=["auc"],
    max_epochs=200,
    patience=20,              # Early stopping
    batch_size=1024,
)
```

!!! note "주요 하이퍼파라미터"
    | 파라미터 | 의미 | 일반적 범위 |
    |---------|------|-----------|
    | `n_steps` | Attention Step 수 | 3~10 |
    | `n_d`, `n_a` | Feature Transformer 차원 | 8~128 |
    | `gamma` | Prior scales 감쇠 (클수록 피처 재사용 허용) | 1.0~2.0 |
    | `lambda_sparse` | Sparsity 정규화 강도 | 1e-4~1e-3 |
    | `mask_type` | Sparsemax 또는 Entmax | `"sparsemax"` |

---

## 참고 자료

[^1]: Sercan O. Arik & Tomas Pfister, "TabNet: Attentive Interpretable Tabular Learning", *AAAI*, 2021. arXiv: [1908.07442](https://arxiv.org/abs/1908.07442). Google Research 제안. 2019년 arXiv 초판, 2021년 AAAI 게재.

[^2]: Leo Grinsztajn, Edouard Oyallon, Gael Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", *NeurIPS Datasets and Benchmarks*, 2022. arXiv: [2207.08815](https://arxiv.org/abs/2207.08815). 45개 데이터셋 벤치마크에서 트리가 DL을 일관되게 앞섬.

[^3]: Ravid Shwartz-Ziv & Amitai Armon, "Tabular Data: Deep Learning is Not All You Need", *Information Fusion*, 2022. arXiv: [2106.03253](https://arxiv.org/abs/2106.03253).

[^4]: Daniel McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", *NeurIPS*, 2023. 176개 데이터셋 메타분석.

[^5]: Niv Kadra et al., "Well-tuned Simple Nets Excel on Tabular Datasets", *NeurIPS*, 2021. 13가지 정규화 조합으로 MLP가 GBDT에 근접하지만 튜닝 비용이 비현실적.

- `pytorch-tabnet` 라이브러리: [github.com/dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet) — TabNet의 PyTorch 구현, pip install pytorch-tabnet으로 설치

!!! tip "다음 섹션"
    TabNet을 살펴보았다면, [CNN · RNN](rnn_tabular.md)에서 CNN/RNN이 정형 데이터에 맞지 않는 이유와 사전 가정(Inductive Bias)의 개념을 다룬다.
