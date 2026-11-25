# TabNet

> Tabular 데이터에 특화된 Attention 기반 신경망 아키텍처.

!!! quote "이 장의 핵심"
    트리 앙상블의 장점인 **피처 선택**, **해석 가능성**, **Sparse한 의사결정**을 신경망 안에 녹여내려 한 것이 TabNet이다. "딥러닝으로 Tabular를 풀겠다"는 시도 중에서, 트리의 작동 원리를 가장 직접적으로 모방한 아키텍처다.

!!! quote "저자의 말"
    솔직히 고백하면, TabNet은 실무에서 써 본 적이 없고 이름만 알던 모형이었다. Sequential Attention, Sparsemax, Self-supervised Pre-training 같은 세부 메커니즘도 이번 가이드북을 쓰면서 처음 제대로 들여다봤다. 그래서 이 장은 실무 경험담이 아니라 **원 논문(Arik & Pfister, AAAI 2021)의 내용을 정리한 것**이다. 수식과 구조 설명은 원 논문 본문과 대조해 확인했고, 벤치마크 결과는 각 논문 원문을 근거로 적었다. 신용평가 데이터에 TabNet을 직접 적용한 결과는 이 장에 포함되어 있지 않다.

---

## 3.1 TabNet이란

### 배경: 왜 Tabular 전용 신경망이 필요한가

[앞 장](lr_as_nn.md)에서 보았듯, Tabular 데이터에서는 MLP가 트리 앙상블에 대체로 밀린다. 원 논문은 그 이유로 기존 신경망 구조(MLP, CNN)가 Tabular의 결정 경계에 맞는 **사전 가정(inductive bias)이 부족**하고 과도하게 파라미터가 많다는 점을 든다. 핵심은 **피처 선택(Feature Selection)** 방식의 차이다.

- **트리**: 각 split에서 **하나의 피처**만 선택하여 분기 → 자연스러운 Sparse 결정
- **MLP**: 모든 뉴런이 **전체 입력**의 가중합을 계산 → Dense 처리, 무의미한 피처에 취약

TabNet(Arik & Pfister, 2021)[^1]은 이 관찰에서 출발한다. 신경망이 **각 의사결정 단계(Step)마다 소수의 피처만 골라서** 사용하도록 만들면, 트리처럼 Sparse하면서도 신경망의 유연성(경사하강 기반 end-to-end 학습)을 유지할 수 있지 않을까?

!!! info "출처"
    TabNet은 Google Cloud AI 소속의 Sercan Ö. Arık과 Tomas Pfister가 2019년 arXiv에 초판을 공개하고, 2021년 AAAI에 게재했다. 논문 제목 자체가 의도를 드러낸다: **"TabNet: Attentive Interpretable Tabular Learning"**, 즉 Attention으로 해석 가능한 Tabular 학습이다.

---

## 3.2 핵심 구조

TabNet의 아키텍처는 **Sequential Attention**을 중심으로 설계되었다. 의사결정을 **여러 단계(Step)**로 나누고, 각 단계마다 **다른 피처 부분집합**을 선택한다.

### 입력

원 논문은 수치형 변수를 별도 전처리 없이 넣고, 범주형 변수는 학습 가능한 임베딩으로 바꿔 넣는다. 전역 정규화(표준화) 대신 입력에 **Batch Normalization**만 적용한다. 배치 크기를 \(B\), 피처 수를 \(D\)라 할 때, BN을 거친 입력 피처 행렬 \(\mathbf{f} \in \mathbb{R}^{B \times D}\)가 모든 Step에 똑같이 들어간다.

### Step-wise 의사결정

TabNet은 \(N_{\text{steps}}\)개의 단계를 순차적으로 수행한다. 각 Step \(i\)에서:

1. **Attentive Transformer**: 직전 Step이 넘겨준 정보 \(\mathbf{a}[i-1]\)로 이번 Step에서 어떤 피처를 볼지 정하는 Sparse mask \(\mathbf{M}[i]\)를 만든다
2. **Feature Transformer**: mask를 곱한 피처 \(\mathbf{M}[i] \cdot \mathbf{f}\)를 처리해, 최종 결정에 쓸 부분 \(\mathbf{d}[i]\)와 다음 Step에 넘길 부분 \(\mathbf{a}[i]\)로 나눈다
3. **집계**: 모든 Step의 \(\mathbf{d}[i]\)에 ReLU를 씌워 더한 뒤, 마지막 선형층으로 출력을 만든다

$$
[\mathbf{d}[i], \mathbf{a}[i]] = f_i(\mathbf{M}[i] \cdot \mathbf{f}), \qquad
\mathbf{d}_{\text{out}} = \sum_{i=1}^{N_{\text{steps}}} \text{ReLU}(\mathbf{d}[i]), \qquad
\hat{\mathbf{y}} = \mathbf{W}_{\text{final}}\, \mathbf{d}_{\text{out}}
\tag{1}
$$

여기서 \(f_i\)는 Step \(i\)의 Feature Transformer, \(\cdot\)은 원소별 곱이다. 분류 문제에서는 \(\hat{\mathbf{y}}\)에 softmax(이진분류라면 sigmoid에 해당)를 씌워 확률을 얻는다. ReLU를 거친 각 Step의 출력을 **더하는** 구조는 결정 트리 여러 개의 출력을 더하는 앙상블을 흉내 낸 것이다.

### Sparse Attention Mask

TabNet의 핵심은 **Sparsemax**로 만든 Attention mask다. Softmax는 모든 피처에 0보다 큰 가중치를 주지만, Sparsemax(Martins & Astudillo, 2016)는 확률 단체(simplex) 위로의 유클리드 사영이어서 상당수 가중치를 **정확히 0**으로 만든다. 이것이 트리의 "일부 피처만 선택"하는 성질을 신경망에서 구현하는 장치다.

$$
\mathbf{M}[i] = \text{sparsemax}\!\left(\mathbf{P}[i-1] \cdot h_i(\mathbf{a}[i-1])\right), \qquad
\sum_{j=1}^{D} M_{b,j}[i] = 1
\tag{2}
$$

\(h_i\)는 학습 가능한 함수(FC층 + BN)다. \(\mathbf{P}[i]\)는 **Prior scale**로, 각 피처가 이전 Step들에서 얼마나 쓰였는지를 누적해 기록한다.

$$
\mathbf{P}[i] = \prod_{j=1}^{i} \left(\gamma - \mathbf{M}[j]\right), \qquad \mathbf{P}[0] = \mathbf{1}
\tag{3}
$$

\(\gamma\)는 완화(relaxation) 파라미터다. \(\gamma = 1\)이면 한 Step에서 크게 쓴 피처는 다음 Step부터 거의 쓸 수 없고, \(\gamma\)가 커질수록 같은 피처를 여러 Step에서 다시 쓸 여지가 커진다. 즉, 각 Step이 **서로 다른 피처에 집중**하도록 유도하는 장치다.

!!! tip "트리 앙상블과의 비유"
    | 트리 앙상블 | TabNet |
    |---|---|
    | 각 노드에서 **1개 피처**로 split | 각 Step에서 **소수의 피처**를 Sparse mask로 선택 |
    | 여러 트리가 서로 다른 피처 조합 사용 | 여러 Step이 서로 다른 피처 조합 사용 (Prior scale로 유도) |
    | 트리 출력을 더해 최종 예측 | Step 출력(ReLU 후)을 더해 최종 예측 |
    | Greedy하게 최적 split 탐색 | 경사하강으로 mask 생성 함수를 학습 |

### Feature Transformer

mask를 곱한 피처는 **Feature Transformer**를 통과한다. 원 논문의 예시 구성은 4개 층으로, 2개는 모든 Step이 **공유**하고 2개는 Step마다 **따로** 둔다. 각 층은 FC → BN → GLU로 이루어지며, 층 사이는 \(\sqrt{0.5}\)를 곱하는 정규화된 잔차 연결로 이어진다. GLU(Gated Linear Unit; Dauphin et al., 2017)는 다음과 같다.

$$
\text{GLU}(\mathbf{u}) = \mathbf{u}_1 \odot \sigma(\mathbf{u}_2)
\tag{4}
$$

입력을 두 부분으로 나누어, 한쪽은 정보를 담고(\(\mathbf{u}_1\)), 다른 쪽은 0~1 사이의 게이트 역할(\(\sigma(\mathbf{u}_2)\))을 한다. LSTM의 게이트와 비슷한 발상이다.

!!! note "Shared vs. Step-specific"
    같은 피처가 여러 Step에 들어가므로, 공통 처리는 공유 층에서 한 번 학습하고 Step별 고유한 처리는 독립 층에서 학습한다. 파라미터를 아끼면서도 표현력을 유지하려는 설계다. 원 논문은 학습 속도를 위해 큰 배치를 쓰고, 입력층을 제외한 BN에는 작은 가상 배치로 나눠 계산하는 Ghost BN을 쓴다.

### Sparsity Regularization

TabNet은 mask가 소수의 피처에 집중하도록 엔트로피 형태의 정규화 항을 손실에 더한다.

$$
L_{\text{sparse}} = \sum_{i=1}^{N_{\text{steps}}} \sum_{b=1}^{B} \sum_{j=1}^{D}
\frac{-M_{b,j}[i] \,\log\!\left(M_{b,j}[i] + \epsilon\right)}{N_{\text{steps}} \cdot B}
\tag{5}
$$

\(\epsilon\)은 수치 안정성을 위한 작은 수다. mask 값이 몇 개 피처에 몰릴수록 엔트로피가 작아진다. 이 항에 계수 \(\lambda_{\text{sparse}}\)를 곱해 전체 손실에 더하며, \(\lambda_{\text{sparse}}\)로 Sparsity 강도를 조절한다.

---

## 3.3 해석 가능성: Instance-wise Feature Importance

TabNet이 "Interpretable"을 제목에 내건 이유는 **mask에서 피처 중요도를 직접 꺼낼 수 있기 때문**이다. \(M_{b,j}[i] = 0\)이면 샘플 \(b\)의 \(j\)번째 피처는 Step \(i\)의 결정에 기여하지 않는다.

### Step별 가중치와 집계 mask

Step마다 최종 결정에 기여한 정도가 다르므로, 원 논문은 샘플 \(b\)의 Step \(i\) 기여 크기를 \(\mathbf{d}[i]\)의 ReLU 합으로 정의한다.

$$
\eta_b[i] = \sum_{c=1}^{N_d} \text{ReLU}\!\left(d_{b,c}[i]\right)
\tag{6}
$$

\(d_{b,c}[i] < 0\)이면 그 Step의 출력이 ReLU에서 잘려 최종 결정에 기여하지 않으므로, 기여가 0이 된다. 이 가중치로 mask를 합산해 **샘플별** 피처 중요도를 만들고, 피처 합이 1이 되도록 정규화한다.

$$
M_{\text{agg},\,b,j} = \frac{\sum_{i=1}^{N_{\text{steps}}} \eta_b[i]\, M_{b,j}[i]}
{\sum_{j=1}^{D} \sum_{i=1}^{N_{\text{steps}}} \eta_b[i]\, M_{b,j}[i]}
\tag{7}
$$

\(\eta_b[i]\)는 Step마다 하나의 상수가 아니라 **샘플마다 다른 값**이다. 그래서 식 (7)은 샘플별(instance-wise) 중요도가 되고, 이를 전체 샘플에 대해 평균하면 전역(global) 중요도가 된다.

### Instance-wise Importance의 의미

샘플별 집계 mask를 보면, **이 고객의 예측에 어떤 변수가 쓰였는지** 개별적으로 볼 수 있다. 모형 구조 안에서 바로 나온다는 점에서 SHAP 같은 사후(post-hoc) 해석 기법과 구별된다.

!!! warning "해석 가능성의 한계"
    TabNet의 mask는 "어떤 피처를 **봤는지**"를 알려줄 뿐, 그 피처가 예측을 **어느 방향으로**(위험 증가 or 감소) 움직였는지는 직접 보여주지 않는다. mask를 거친 피처가 Feature Transformer의 비선형 변환을 통과하기 때문이다. 거절 사유처럼 방향이 필요한 설명에는 여전히 SHAP 등 사후 분석이 필요하다.

    트리 앙상블의 split 기반 Feature Importance도 같은 한계를 공유한다. 중요도는 알 수 있지만 방향(+/−)은 별도 분석이 필요하다. 이 점에서 TabNet의 해석 가능성이 트리보다 "더 좋다"고 단정하기는 어렵다.

---

## 3.4 Self-supervised Pre-training

TabNet의 또 다른 특징은 **Self-supervised Pre-training**이다.

라벨이 없는 데이터로 먼저 TabNet 인코더를 학습시킨 뒤, 라벨이 있는 데이터로 Fine-tuning한다. 사전학습 과제는 **가려진 피처 값 예측**이다. 샘플마다 일부 피처를 무작위로 가린 이진 mask \(\mathbf{S}\)를 만들고, 인코더에는 가려지지 않은 피처만 넣은 뒤, 디코더가 가려진 피처의 값을 복원하도록 학습한다. 복원 오차는 피처마다 척도가 다르므로 각 피처의 표준편차로 나눠 정규화한다.

!!! example "신용평가에서의 의미"
    신용평가 데이터는 라벨(부도 여부) 확정까지 **성과 관찰 기간**(보통 12~24개월)이 필요하다. 라벨이 아직 확정되지 않은 최근 신청 데이터, 그리고 거절되어 성과를 관찰할 수 없는 신청자 데이터는 풍부하지만, 라벨이 있는 데이터는 제한적이다.

    Self-supervised Pre-training은 라벨 없는 데이터로 피처 간 관계를 먼저 학습하고, 라벨 데이터로 Fine-tuning하는 전략을 가능하게 한다. 원 논문은 라벨 없는 데이터가 풍부할 때 사전학습이 성능을 크게 끌어올린다고 보고한다. 다만 이것이 거절자 편향을 해결해 주는 것은 아니다. 거절자의 성과를 추정하는 문제는 스코어카드의 거절추론(Reject Inference)과 별개로 다뤄야 한다.

---

## 3.5 신용평가에서의 위치

### GBDT와의 비교

| | TabNet | GBDT (XGBoost/LightGBM) |
|---|---|---|
| **피처 선택** | Attention mask (학습 기반) | Split 기반 (Greedy) |
| **해석 가능성** | mask → 피처 중요도 (방향 없음) | Split Importance, SHAP |
| **결측치 처리** | 원 논문에 별도 메커니즘 없음 → 사전 대체 필요 | 자체 처리 |
| **범주형 변수** | 학습 가능한 임베딩 (구조에 내장) | 자체 처리(LightGBM, CatBoost 등) 또는 인코딩 |
| **하이퍼파라미터** | 다수 (Steps, mask 차원, \(\gamma\), \(\lambda_{\text{sparse}}\), 학습률, 배치 크기 등) | 상대적으로 적음 |
| **학습 속도** | 느림 (GPU 권장) | 빠름 (CPU로 충분) |
| **재현성** | GPU 비결정성 이슈 | 상대적으로 확보 쉬움 (옵션·버전·환경 고정 필요) |
| **Pre-training** | 지원 (가려진 피처 복원) | 미지원 |

### 벤치마크 결과

TabNet 원 논문은 Forest Cover Type, Poker Hand 등 여러 데이터셋에서 XGBoost·LightGBM·CatBoost보다 좋은 결과를 보고했다. 그러나 이후의 독립 벤치마크는 다른 그림을 보여준다.

- **Shwartz-Ziv & Armon (2022)**[^2]: TabNet을 포함한 딥러닝 모형 4종을 XGBoost와 비교했다. TabNet은 **원 논문에 쓰인 데이터셋**에서는 XGBoost와 경쟁적이었지만, 그 밖의 데이터셋에서는 대부분 XGBoost에 뒤졌다. 원 논문에 쓰이지 않은 데이터셋에서 최고 성능 대비 평균 상대 성능 저하는 TabNet 약 10.5%, XGBoost 약 3.3%였다. 저자들은 각 딥러닝 모형이 자기 논문의 데이터셋에서만 우세했다고 정리했다.
- **Grinsztajn et al. (2022)**[^3]: TabNet은 비교 대상에 포함되지 않았다. 다만 MLP, ResNet, FT-Transformer, SAINT 같은 딥러닝 모형 전반이 45개 데이터셋에서 트리 기반 모형에 밀렸다.
- **McElfresh et al. (2023)**[^4]: 176개 데이터셋에서 19개 알고리즘을 비교했다. "신경망 vs GBDT" 논쟁 자체가 과장되었다는 결론이지만, 불규칙한 변수 분포를 가진 데이터에서는 GBDT가 뚜렷이 강했다.

!!! note "공정한 평가"
    TabNet이 트리에 밀리는 이유 중 하나는 **튜닝 난이도**다. GBDT는 기본 하이퍼파라미터로도 합리적인 성능을 내지만, TabNet은 \(N_{\text{steps}}\), \(N_a\), \(N_d\), \(\gamma\), \(\lambda_{\text{sparse}}\), 학습률, 배치 크기 등 조정할 파라미터가 많다. 반대 방향의 연구도 있다. Kadra et al.(2021)[^5]은 13가지 정규화 기법의 조합을 데이터셋마다 탐색해 튜닝한 단순 MLP가 TabNet 같은 특화 구조는 물론 XGBoost보다도 좋았다고 보고했다. 신경망의 성패가 **구조보다 튜닝에** 달려 있다는 점을 보여 주는 결과이며, 그만큼 튜닝 비용을 감수해야 한다는 뜻이기도 하다.

### 실무 관점

신용평가 실무에서 TabNet의 입지는 다음과 같이 정리된다.

1. **XGBoost/LightGBM이 기본 선택지**: Tabular 데이터에서 성능, 속도, 재현성, 해석 도구 모든 면에서 검증됨
2. **TabNet은 특정 조건에서 고려**: 라벨이 부족하고 라벨 없는 데이터가 많은 상황에서 Self-supervised Pre-training이 유리할 수 있음
3. **해석 가능성은 기대보다 제한적**: mask만으로는 기여 방향을 알 수 없어 거절 사유 산출 등 규제 요건을 충족하기 어려움. SHAP 등 추가 분석이 필요

!!! tip "그래서 언제 써야 하나?"
    현실적으로 TabNet이 GBDT를 대체할 명확한 우위 시나리오는 좁다. 하지만 **"딥러닝으로 Tabular를 풀려면 어떤 구조가 필요한가"**를 이해하는 데는 매우 교육적인 아키텍처다.

    - 트리의 Sparse split → Sparsemax Attention
    - 단계적 피처 선택 → Sequential multi-step 구조
    - 앙상블의 출력 합산 → Step별 출력의 합산

    TabNet은 "트리가 왜 Tabular에서 강한가"에 대한 답을 **신경망 언어로 번역한 것**이다. 트리가 이기는 이유를 이해하면, TabNet의 설계 동기도 자연스럽게 이해된다.

---

## 3.6 주요 하이퍼파라미터

TabNet의 대표적인 오픈소스 구현은 `pytorch-tabnet` 라이브러리다. 아래 기본값과 범위는 이 라이브러리 문서 기준이다.

| 파라미터 | 의미 | 기본값 | 문서의 권장 범위 |
|---------|------|:---:|-----------|
| `n_steps` | 의사결정 Step 수 \(N_{\text{steps}}\) | 3 | 3~10 |
| `n_d`, `n_a` | 결정용 출력 \(\mathbf{d}\)와 다음 Step용 출력 \(\mathbf{a}\)의 차원 | 8, 8 | `n_d`는 8~64, 보통 `n_d` = `n_a` |
| `gamma` | Prior scale의 완화 계수 \(\gamma\) (클수록 피처 재사용 허용) | 1.3 | 1.0~2.0 |
| `lambda_sparse` | Sparsity 정규화 강도 \(\lambda_{\text{sparse}}\) | 1e-3 | 문제에 따라 줄이면 도움이 될 수 있음 |
| `mask_type` | mask 함수 | sparsemax | sparsemax 또는 entmax |

`mask_type`의 entmax는 Softmax와 Sparsemax의 중간 성질을 가진 함수로, Sparsemax보다 덜 극단적으로 0을 만든다.

---

## 참고 자료

[^1]: Sercan Ö. Arık & Tomas Pfister, "TabNet: Attentive Interpretable Tabular Learning", *AAAI*, 2021. arXiv: [1908.07442](https://arxiv.org/abs/1908.07442). 2019년 arXiv 초판, 2021년 AAAI 게재.

[^2]: Ravid Shwartz-Ziv & Amitai Armon, "Tabular Data: Deep Learning is Not All You Need", *Information Fusion*, vol. 81, 2022. arXiv: [2106.03253](https://arxiv.org/abs/2106.03253).

[^3]: Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", *NeurIPS Datasets and Benchmarks*, 2022. arXiv: [2207.08815](https://arxiv.org/abs/2207.08815).

[^4]: Duncan McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", *NeurIPS Datasets and Benchmarks*, 2023. arXiv: [2305.02997](https://arxiv.org/abs/2305.02997).

[^5]: Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka, "Well-tuned Simple Nets Excel on Tabular Datasets", *NeurIPS*, 2021. arXiv: [2106.11189](https://arxiv.org/abs/2106.11189).

- Martins & Astudillo, "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification", *ICML*, 2016.
- Dauphin et al., "Language Modeling with Gated Convolutional Networks", *ICML*, 2017. — GLU
- `pytorch-tabnet` 라이브러리: [github.com/dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)

!!! tip "다음 섹션"
    TabNet을 살펴보았다면, [CNN · RNN](rnn_tabular.md)에서 CNN/RNN이 정형 데이터에 맞지 않는 이유와 사전 가정(Inductive Bias)의 개념을 다룬다.
