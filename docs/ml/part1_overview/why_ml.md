# 왜 ML인가

## 1.1 전통 스코어카드가 잘한 것

머신러닝을 이야기하기 전에, 먼저 전통 스코어카드가 **왜 수십 년간 살아남았는지**를 인정해야 한다.

| 강점 | 설명 |
|------|------|
| **완벽한 해석 가능성** | 구간별 배점(β × WoE를 점수 척도로 변환한 값)의 합 = 점수. 점수표 한 장으로 "왜 이 고객이 이 등급인지" 즉시 설명 가능 |
| **규제 수용성** | 금감원·바젤 프레임워크가 요구하는 투명성·감사 추적을 자연스럽게 충족 |
| **운영 안정성** | 모형 구조가 단순하여 배포·모니터링·재개발이 쉬움 |
| **실무자 이해도** | 현업 심사역이 점수표를 보고 판단 근거를 이해할 수 있음 |

로지스틱 회귀 + WoE 조합은 신용평가의 **기본 문법**이다. 이것이 나쁜 모형이어서 ML로 가는 것이 아니다.

> 전통 스코어카드는 "나쁜 모형"이 아니라, **"더 잘할 수 있는 여지"가 있는 모형**이다.

---

## 1.2 그런데 왜 부족한가

전통 스코어카드의 구조적 한계는 세 가지로 압축된다.

### 한계 ① — 선형 관계만 포착한다

로지스틱 회귀는 log-odds를 입력 변수의 **선형 결합**으로 모형화한다.

$$
\ln\frac{p}{1-p} = \beta_0 + \beta_1 \cdot \text{WoE}_1 + \beta_2 \cdot \text{WoE}_2 + \cdots
$$

WoE 변환이 비선형성을 **간접적으로** 반영하긴 하지만, 그것은 변수 하나 안에서의 이야기다. 변수들이 결합되는 방식은 여전히 선형이므로, 복잡한 비선형 패턴은 구조적으로 포착할 수 없다.

<figure markdown="span">
  ![선형 vs 비선형 Decision Boundary](images/linear_vs_nonlinear_boundary.png){ width="100%" }
  <figcaption>가상 데이터. 소득이 낮을수록, DSR이 높을수록 불량이 많고, 소득 5,000만 원을 경계로 위험이 급등하는 DSR 임계값이 40%와 60%로 다르게 설정되어 있다. 직선 하나로 나누는 로지스틱 회귀는 이 계단 구조를 표현하지 못하지만, 얕은 트리(depth 2)를 쌓은 Gradient Boosting은 비선형 경계를 학습한다. 두 모형 모두 검증셋 AR이 학습셋 AR과 크게 다르지 않아, 경계가 노이즈를 외운 결과가 아님을 함께 보여 준다.</figcaption>
</figure>

!!! example "예시: DSR의 비선형 효과"
    DSR(총부채원리금상환비율) 30% 이하에서는 연체율 차이가 거의 없다가, 40%를 넘기면 급격히 상승하고, 80% 이상에서는 다시 둔화되는 **S자 형태**의 패턴이 있다고 하자. WoE 구간화로 이를 **근사**할 수는 있지만, 구간 경계의 선택에 따라 정보 손실이 발생하고, 변수가 수십 개일 때 각각을 최적 구간화하는 것은 현실적으로 어렵다.

### 한계 ② — 변수 간 상호작용을 반영하지 못한다

현실의 신용 리스크는 변수 하나로 결정되지 않는다.

- **소득이 높고** DSR이 낮으면 → 안전
- **소득이 높지만** DSR도 높으면 → 위험할 수 있음
- **소득이 낮지만** DSR도 낮으면 → 보수적이지만 안정적

이런 **교호작용(interaction)**은 두 변수의 조합에서 나오는 패턴인데, 로지스틱 회귀의 선형 결합 구조에서는 각 변수의 효과가 독립적으로 합산될 뿐, 조합에 따른 차별적 효과를 자동으로 학습하지 못한다.

물론 \(\beta_3 \cdot X_1 \cdot X_2\) 같은 교호작용 항을 수동으로 추가할 수는 있다. 그러나 변수가 50개일 때 2차 교호작용만 해도 \(\binom{50}{2} = 1{,}225\)개 — 이것을 사람이 일일이 탐색하고 선별하는 것은 비현실적이다.

### 한계 ③ — 수작업 WoE Classing의 비효율

전통 스코어카드 개발 과정에서 가장 시간이 많이 소요되는 단계가 **Fine Classing → Coarse Classing → WoE 변환**이다.

1. 변수마다 구간을 나누고
2. 단조성(monotonicity)을 검증하고
3. 구간 수를 조정하며
4. IV 기준으로 변수를 선별한다

이 과정을 수십 개 변수에 대해 반복한다. OptBinning 같은 자동화 도구가 있지만, 최종 검토와 조정은 여전히 사람의 판단에 의존한다.

트리 기반 모형은 이 전처리 과정이 **구조적으로 불필요**하다. 트리의 분할(split) 자체가 최적의 구간을 자동으로 찾아주기 때문이다.

---

## 1.3 ML이 가져온 변화

위 세 가지 한계를 트리 기반 ML 모형이 어떻게 해결하는지 대응시켜 보자.

| 전통 스코어카드의 한계 | 트리 앙상블의 해법 |
|----------------------|-------------------|
| 선형 관계만 포착 | 트리의 재귀적 분할로 **임의의 비선형 패턴** 자동 학습 |
| 교호작용 불가 | 트리 깊이 ≥ 2이면 **자동으로 교호작용 포착** |
| 수작업 WoE Classing | 트리 split이 **최적 구간을 자동 탐색** — 전처리 부담 대폭 감소 |

### 비선형 패턴의 자동 학습

트리는 변수의 값을 기준점(threshold)에서 나누는 구조이므로, 계단 함수(piecewise-constant)를 자연스럽게 만든다. 앙상블로 수백 개의 트리를 합치면, 이 계단 함수들이 중첩되어 **사실상 임의의 비선형 곡선**을 근사할 수 있다.

WoE 구간화도 결국 같은 원리(piecewise-constant 변환)이지만, 트리 앙상블은 이것을 **모든 변수에 대해 동시에, 자동으로** 수행한다는 차이가 있다.

### 교호작용의 자동 포착

트리에서 depth ≥ 2이면, 첫 번째 분할의 결과에 따라 두 번째 분할이 달라진다. 이것이 곧 교호작용이다.

```
                 [소득 ≥ 5,000만?]
               No /            \ Yes
      [DSR > 40%?]              [DSR > 60%?]
     No /      \ Yes           No /      \ Yes
  Bad 3%     Bad 15%       Bad 1.5%    Bad 9%
```

(가상 수치. 왼쪽 가지 = No, 오른쪽 가지 = Yes)

소득이 낮은 그룹에서는 DSR 40%가 기준이고, 높은 그룹에서는 DSR 60%가 기준이다. 같은 DSR 변수에 대해 **소득 수준에 따라 다른 임계값**이 적용된다. 같은 DSR 구간이라도 고소득 그룹의 불량률이 더 낮다는 점도 함께 표현된다. 이것이 사람이 수동으로 만들어야 했던 교호작용을 트리가 자동으로 학습한 결과다. 위 그림의 가상 데이터도 이 구조로 만들었다.

!!! info "Depth = 1이면 교호작용이 사라진다"
    Depth = 1인 트리(stump)를 사용하면 각 트리가 변수 하나만 사용하므로 교호작용이 구조적으로 불가능하다. 이 구조는 [1-Depth GBM 스코어카드](../part4_interpretation/depth1_gbm.md)에서 다룬다.

### 변수 전처리의 간소화

| 전처리 단계 | 전통 스코어카드 | 트리 앙상블 |
|------------|---------------|------------|
| 결측치 처리 | 별도 구간화 필요 | XGBoost/LightGBM이 자체 처리 |
| 이상치 처리 | 구간 경계 조정 필요 | 트리 split이 자연스럽게 분리 |
| 구간화 (Classing) | 수작업 Fine/Coarse Classing | **불필요** — split이 자동 수행 |
| 단조성 검증 | 수동 확인 필수 | 모형이 자동 학습 (또는 monotone constraint 설정) |
| 변수 선택 | IV 기준 수작업 선별 | Feature Importance / SHAP 기반 자동화 |

---

## 1.4 왜 하필 트리 앙상블인가

ML 알고리즘은 SVM, 신경망, 커널 방법 등 다양하다. 그런데 왜 **신용평가를 포함한 tabular 데이터 영역**에서는 트리 앙상블(특히 Boosting)이 압도적인 선택인가?

이 질문에 대한 학술적 답변은 2021년 무렵부터 나온 일련의 대규모 벤치마크 연구에서 찾을 수 있다.

### Grinsztajn et al. (NeurIPS 2022)

> *"Why do tree-based models still outperform deep learning on tabular data?"*

45개 tabular 데이터셋(학습셋을 1만 건 수준으로 맞춘 중간 규모 설정)에서 트리 기반 모델(RF, GBT, XGBoost)과 딥러닝 모델(MLP, ResNet, FT-Transformer, SAINT)을 체계적으로 비교한 이 논문은, 트리가 우세한 **세 가지 이유**를 실험으로 확인했다.

| 이유 | 설명 |
|------|------|
| **신경망은 지나치게 매끄러운 해를 선호한다** | Tabular 데이터의 타겟 함수는 부드럽게 연결되지 않고, 특정 임계점에서 급격히 변하는 패턴이 흔하다. 타겟을 인위적으로 매끄럽게 만들면 트리의 성능은 떨어지지만 신경망은 거의 영향을 받지 않았다. 신경망의 사전 가정(Inductive Bias)이 매끄러운 함수 쪽에 있어 불규칙한 패턴에 불리하다는 뜻이다. 트리는 piecewise-constant 함수를 만들므로 이런 편향이 없다 |
| **무의미한 변수에 약하다** | 트리의 split은 변별력이 없는 변수를 자연스럽게 무시한다. MLP 계열은 모든 입력을 가중합하므로, 무의미한 변수를 추가하면 트리와의 격차가 벌어지고 제거하면 좁혀졌다 |
| **회전 불변성이 오히려 해가 된다** | Tabular 데이터의 각 컬럼은 독립적인 비즈니스 의미를 가진다(소득, 연체 횟수, DSR 등). MLP는 입력 축을 회전해도 같은 방식으로 학습하는 구조(회전 불변)라 원래 변수 축의 의미를 활용하지 못한다. 트리는 "소득 > 3,000만?"처럼 **변수 하나의 기준점**(축 정렬 분할)으로 자르므로 이 구조에 맞는다. 데이터를 무작위로 회전시키자 순위가 뒤집혀 신경망이 트리를 앞섰다 |

### 후속 연구들

| 논문 | 핵심 결론 |
|------|----------|
| **Shwartz-Ziv & Armon (Information Fusion, 2022)** | TabNet·NODE·DNF-Net·1D-CNN을 XGBoost와 비교. 각 딥러닝 모형은 **자기 논문에 쓰인 데이터셋에서만** 우세했고, 그 밖의 데이터셋에서는 XGBoost가 대체로 앞섰으며 튜닝 부담도 훨씬 작았다. 딥러닝과 XGBoost의 앙상블은 XGBoost 단독보다 나았다 |
| **Kadra et al. (NeurIPS 2021)** | 반론에 해당하는 연구. 13가지 정규화 기법의 조합("regularization cocktail")을 데이터셋마다 탐색해 튜닝한 단순 MLP가 40개 데이터셋에서 특화 신경망 구조는 물론 XGBoost보다도 우수했다고 보고했다. 다만 데이터셋마다 정규화 조합과 하이퍼파라미터를 대규모로 탐색해야 하므로 튜닝 비용이 크다 |
| **McElfresh et al. (NeurIPS 2023)** | 176개 데이터셋, 19개 알고리즘 비교. "신경망 vs GBDT" 논쟁은 과장되어 있으며, 많은 데이터셋에서 차이가 작거나 기본 모형을 가볍게 튜닝하는 것이 알고리즘 선택보다 중요하다고 결론지었다. 다만 치우치거나 꼬리가 두꺼운 변수 분포처럼 불규칙한 데이터에서는 GBDT가 뚜렷이 강했다 |
| **Borisov et al. (IEEE TNNLS, 2024)** | 딥러닝 기반 tabular 방법론 서베이. 지도학습 과제에서 gradient-boosted tree 앙상블이 여전히 대부분 딥러닝 모형을 앞선다고 정리했다 |

정리하면, "트리가 언제나 이긴다"는 법칙은 아니다. 다만 신용평가처럼 불규칙한 임계 패턴, 치우친 금액 변수, 다수의 약한 변수가 섞인 정형 데이터에서는 **튜닝 비용 대비 성능**까지 고려할 때 GBDT가 가장 안전한 기본 선택이라는 데 여러 연구가 대체로 일치한다. 저자도 LSTM 등 딥러닝 구조를 신용평가 데이터에 시험해 보았지만 트리 앙상블 대비 이득을 얻지 못했다([CNN · RNN](../part2_neural_net/rnn_tabular.md) 참조).

!!! warning "딥러닝이 나쁘다는 뜻이 아니다"
    이미지, 텍스트, 음성 같은 **비정형 데이터**에서는 딥러닝이 압도적이다. 위 결론은 tabular(정형) 데이터에 한정된 이야기다. Tabular 데이터의 특성(이질적 피처 타입, 축 정렬 구조, 상대적으로 작은 데이터 크기)이 트리 앙상블에 유리하게 작용하는 것이다. 최근에는 소규모 tabular 데이터(약 1만 건 이하)에서 GBDT를 앞선다고 보고된 사전학습형 모형 TabPFN(Hollmann et al., *Nature*, 2025)처럼 새로운 흐름도 있으나, 신용평가 규모의 데이터와 운영 환경에서의 검증은 아직 제한적이다.

### 신용평가 데이터는 전형적인 Tabular

신용평가 데이터는 위 논문들이 말하는 "트리에 유리한 조건"을 거의 완벽하게 충족한다.

- **다양한 변수 타입**: 연속형(소득, DSR), 범주형(대출 목적, 금융기관 유형), 여부 변수(연체 유무)가 혼재
- **변수 단위 분할에 적합**: 각 컬럼이 독립적 의미를 가져, 트리의 "변수 하나씩 자르기"와 잘 맞음
- **Irregular한 타겟**: 연체율이 특정 임계점(DSR 40%, 연체 이력 유무 등)에서 급변
- **노이즈 피처 다수**: 수백 개 후보 변수 중 실제 유효한 것은 일부
- **중소규모 데이터**: 수만~수십만 건 (딥러닝의 데이터 효율성이 떨어지는 구간)

---

## 1.5 규제 환경과 현실적 제약

ML의 예측 성능이 아무리 뛰어나도, **규제 기관이 수용하지 않으면** 실전에 투입할 수 없다.

### 국내 규제 환경

한국에서 신용평가모형은 여러 층위의 규율을 받는다. 법률인 「신용정보의 이용 및 보호에 관한 법률」(신용정보법), 그 하위의 금융위원회 고시인 신용정보업감독규정, 금융위원회의 금융분야 AI 가이드라인, 그리고 2026년 1월 시행된 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」(인공지능 기본법)이 있다. ML 관점에서 특히 문제가 되는 요건은 다음과 같다.

| 요건 | 근거 | ML 관점에서의 도전 |
|------|------|-------------------|
| **자동화평가 설명·이의제기** | 신용정보법 제36조의2: 개인은 자동화평가 여부, 평가 결과, 주요 기준, 기초정보의 개요 등에 대한 설명을 요구하고, 기초정보 정정·평가 재산출 등을 요구할 수 있다 | 수백 개 트리의 합산에서 "주요 기준"을 뽑아내려면 SHAP 등 해석 도구가 필요 |
| **모형 투명성·검증** | 감독규정과 AI 가이드라인의 모형 관리 요건 | 구조를 직접 읽을 수 없으므로 문서화·검증 부담이 커짐 |
| **공정성** | AI 가이드라인의 공정성 원칙 | 민감 변수를 빼도 대리(proxy) 변수를 통한 간접 차별을 점검해야 함 |
| **고영향 AI 의무** | 인공지능 기본법: 대출 심사처럼 개인의 권리·의무에 중대한 영향을 주는 판단에 쓰이는 AI는 고영향 AI로 분류될 수 있다 | 위험관리·설명 방안·인적 관리감독 등 사업자 의무 대응 |

각 규제의 상세와 ML 관점의 대응은 [규제 프레임워크](../part5_validation/regulatory.md)에서 정리한다.

### 현재의 현실: 하이브리드 접근

이러한 규제 환경 때문에, 국내 CB사 및 금융기관에서는 다음과 같은 **점진적 도입** 전략을 취하고 있다.

**전통 스코어카드가 여전히 주력인 영역:**

- 금감원 검사 대상인 **규제모형** — CB사의 개인신용평점, 금융기관의 내부 심사모형 등
- 소비자 대면 심사 — 거절 사유 설명 의무가 있는 영역

**ML이 활발히 적용되는 영역:**

- **전략모형(Strategy Model)** — 규제모형(CSS/BSS)의 등급과 ML 모형의 스코어를 **Cross Matrix**로 결합하여, 심사·한도·금리 등 의사결정을 오버라이드하는 용도. 규제모형 자체를 대체하지 않으면서 ML의 예측력을 활용하는 가장 현실적인 방법이다
- 사기 탐지(FDS) — 패턴의 복잡성이 높고, 해석보다 탐지 정확도가 우선
- 마케팅 타겟팅 — 규제 요건이 상대적으로 낮음
- 내부 리스크 관리 — 규제모형 외 보조적 의사결정
- 조기경보 시스템 — 연체 전조를 포착하는 모니터링용

!!! example "Cross Matrix 오버라이드 예시"
    규제모형(CSS) 등급과 ML 스코어를 2차원으로 교차시켜, 셀별로 차별화된 의사결정을 적용한다.

    | | ML 상위 (Low Risk) | ML 중위 | ML 하위 (High Risk) |
    |---|:---:|:---:|:---:|
    | **CSS 1~3등급** | 자동 승인 | 자동 승인 | 서류 심사 |
    | **CSS 4~6등급** | 한도 상향 | 기본 한도 | 한도 하향 |
    | **CSS 7~10등급** | 재심사 | 거절 | 거절 |

    CSS만으로는 동일 등급 내 리스크 차이를 구분할 수 없지만, ML 스코어를 교차하면 **같은 등급 안에서도** 승인/거절/한도를 세분화할 수 있다. 규제모형의 프레임은 유지하면서 ML의 변별력을 얹는 구조다.

**전통 + ML 하이브리드:**

- 전통 스코어카드의 변수 선정 단계에서 SHAP/Feature Importance 활용
- 전통 모형을 주 모형, ML을 챌린저 모형(challenger model)으로 병행 운영

!!! tip "흐름은 ML 쪽으로"
    규제가 ML을 막는 것은 아니다 — **해석 가능한 형태**로 도입하라는 것이다. SHAP 등 사후 해석 도구의 발전으로, 규제모형에서도 ML 도입의 문턱은 점차 낮아지고 있다.

---

## 1.6 이 섹션의 로드맵

이 머신러닝 섹션은 전통 스코어카드의 기초를 전제로, 트리 기반 ML 모형을 **이론부터 실전까지** 따라갈 수 있도록 구성했다.

| 파트 | 주제 | 핵심 내용 |
|:---:|------|----------|
| **1. 개요** | [왜 ML인가](why_ml.md) (현재 페이지) | 전통 → ML 전환의 동기와 맥락 |
| | [Bias-Variance Tradeoff](bias_variance.md) | 모형 설계의 근본 원리 |
| | [정규화](regularization.md) | Ridge, Lasso, Shrinkage — 과적합 제어의 수학적 도구 |
| | [Baseline 워크플로우](baseline_workflow.md) | 단순 모형 기준선으로 추가 복잡도의 값어치 판단 |
| | [데이터 분리 전략](data_splitting.md) | Holdout·OOT, 층화추출, CV |
| | [피처 전처리](feature_preprocessing.md) · [피처 선택](feature_selection.md) | WoE 없는 변수 처리, SHAP/Boruta 선택 |
| **2. 뉴럴넷** | [신경망 기초](../part2_neural_net/nn_basics.md) · [LR = 단일 뉴런](../part2_neural_net/lr_as_nn.md) | 퍼셉트론, LR과 NN의 연결 |
| | [TabNet](../part2_neural_net/tabnet.md) · [CNN · RNN](../part2_neural_net/rnn_tabular.md) | Tabular 특화 신경망, 사전 가정(Inductive Bias)과 LSTM 실무 경험 |
| **3. 트리 앙상블** | [CART](../part3_tree_ensemble/tree_models.md) → [RF](../part3_tree_ensemble/ensemble.md) → [트리의 Bias-Variance](../part3_tree_ensemble/bias_variance_trees.md) → [Boosting 기초](../part3_tree_ensemble/boosting_fundamentals.md) · [심화](../part3_tree_ensemble/boosting_advanced.md) → [XGBoost/LightGBM](../part3_tree_ensemble/xgb_lgbm.md) → [하이퍼파라미터 튜닝](../part3_tree_ensemble/hyperparameter_tuning.md) | 트리에서 실무 표준 도구까지 |
| **4. 해석과 설명** | [해석 가능성](../part4_interpretation/interpretability.md) · [SHAP 이론](../part4_interpretation/shap_theory.md) | SHAP, PDP, TreeSHAP |
| | [1-Depth GBM 스코어카드](../part4_interpretation/depth1_gbm.md) · [EBM (GA²M)](../part4_interpretation/ebm.md) | 해석 가능한 구조의 부스팅 모형 |
| | [fANOVA와 Purification](../part4_interpretation/fanova_concepts.md) · [도구 · 구현](../part4_interpretation/depth1_tools_impl.md) · [해석 고민 기록](../part4_interpretation/shap_in_practice.md) | 효과 분리, 실무 적용 |
| **5. 모델 검증** | [모델 검증](../part5_validation/validation.md) · [규제 프레임워크](../part5_validation/regulatory.md) | KS/AR, 캘리브레이션, OOT, 규제 대응 |

## 참고 자료

- Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", *NeurIPS Datasets and Benchmarks*, 2022. arXiv: [2207.08815](https://arxiv.org/abs/2207.08815)
- Shwartz-Ziv & Armon, "Tabular Data: Deep Learning is Not All You Need", *Information Fusion*, 81, 2022. arXiv: [2106.03253](https://arxiv.org/abs/2106.03253)
- Kadra, Lindauer, Hutter & Grabocka, "Well-tuned Simple Nets Excel on Tabular Datasets", *NeurIPS*, 2021. arXiv: [2106.11189](https://arxiv.org/abs/2106.11189)
- McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", *NeurIPS Datasets and Benchmarks*, 2023. arXiv: [2305.02997](https://arxiv.org/abs/2305.02997)
- Borisov et al., "Deep Neural Networks and Tabular Data: A Survey", *IEEE Transactions on Neural Networks and Learning Systems*, 35(6), 2024. arXiv: [2110.01889](https://arxiv.org/abs/2110.01889)
- Hollmann et al., "Accurate predictions on small data with a tabular foundation model", *Nature*, 637, 2025.

!!! tip "다음 섹션"
    이 섹션의 여정은 [Bias-Variance Tradeoff](bias_variance.md)에서 시작한다. "멍청한 모형"과 "똑똑한 모형"의 차이가 무엇인지, 그리고 모형 튜닝의 본질이 무엇인지를 먼저 잡아야 이후 모든 알고리즘이 제자리에 놓인다.
