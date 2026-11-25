# CNN · RNN — 정형 데이터에서의 한계

> CNN과 RNN은 각각 이미지와 시계열에 특화된 아키텍처다. 신용평가 정형 데이터에는 맞지 않지만, 왜 안 맞는지를 이해하면 딥러닝의 **사전 가정(Inductive Bias)**이 보인다.

!!! quote "이 장의 핵심"
    모든 신경망 아키텍처에는 **"이 데이터는 이런 구조를 가지고 있을 것이다"**라는 가정이 내장되어 있다. CNN은 공간적 지역성, RNN은 시간적 순서를 가정한다. 정형 데이터는 이 두 가정 모두에 해당하지 않는다.

---

## 4.1 사전 가정 (Inductive Bias)

딥러닝 아키텍처를 이해하는 열쇠는 **사전 가정**이다.

| 아키텍처 | 사전 가정 | 의미 |
|---------|-----------|------|
| **MLP** | 구조적 가정은 약함. 다만 **매끄러운 함수 선호**, **회전 불변성** | 입력 간 관계는 정해 두지 않지만, 경사하강으로 학습한 해가 매끄러운 쪽으로 치우치고, 입력 축을 회전해도 같은 방식으로 학습 |
| **CNN** | **공간적 지역성** + **이동 등변성** | 가까운 픽셀이 먼 픽셀보다 관련이 깊고, 같은 패턴은 위치가 달라도 같은 필터로 탐지 |
| **RNN** | **시간적 순서** + **순차적 의존성** | 과거 정보가 현재에 영향을 주고, 순서가 중요 |
| **Transformer** | **순서 무관(순열 등변성)** + 모든 원소 간 관계 학습 | 입력 원소의 순서를 가정하지 않고 Attention으로 관계를 학습. 순서가 필요한 데이터에는 Position Encoding으로 순서를 주입 |

정형(Tabular) 데이터의 특성:

- **컬럼 간 순서가 없다** — "DSR"이 "연체횟수" 옆에 있든 멀리 있든 의미가 같다
- **공간적 지역성이 없다** — 인접 컬럼끼리 특별히 관련이 깊지 않다
- **시계열 구조가 아니다** — 각 행(고객)은 관측 시점 기준의 스냅샷이다(과거 이력은 집계 변수로 들어온다)

!!! info "왜 MLP도 Tabular에서 트리에 밀리나?"
    MLP는 CNN·RNN처럼 강한 구조적 가정이 없으므로, Tabular에 "맞지 않는 가정을 강요하는" 문제는 없다. 문제는 MLP가 가진 약한 가정들이 Tabular와 어긋난다는 점이다. Grinsztajn et al.(2022)은 MLP 계열이 **지나치게 매끄러운 해를 선호**하고, **무의미한 변수에 약하며**, 입력 축의 의미를 활용하지 못하는 **회전 불변성**을 가진다는 점을 트리에 밀리는 이유로 꼽았다([왜 ML인가](../part1_overview/why_ml.md) 참조). 반대로 트리는 "변수 하나의 기준점으로 자르는 축 정렬 분할(axis-aligned split)"이라는, Tabular에 잘 맞는 사전 가정을 가지고 있다. 트리 자체는 [트리 앙상블](../part3_tree_ensemble/index.md) 장에서 본격적으로 다룬다.

    Transformer의 "순서 무관" 가정은 컬럼 순서가 의미 없는 Tabular와 오히려 잘 맞는다. FT-Transformer 같은 Tabular용 Transformer가 시도되는 이유다.

---

## 4.2 CNN — 왜 정형 데이터에 안 맞는가

CNN(Convolutional Neural Network)은 **합성곱 필터**가 입력의 지역적 패턴을 탐지하고, 이 필터를 전체 입력에 걸쳐 공유(weight sharing)하는 구조다.

### CNN의 핵심 가정

1. **지역성(Locality):** 의미 있는 패턴은 인접한 요소들의 조합에서 나타남
2. **이동 등변성(Translation Equivariance):** 같은 패턴은 어디에 나타나든 같은 필터로 탐지됨. 입력이 이동하면 탐지 결과도 그만큼 이동한다(pooling을 거치면 위치에 덜 민감한 근사적 불변성이 된다)

이미지에서 이 가정은 잘 들어맞는다. 고양이 귀의 패턴은 이미지 왼쪽 상단이든 오른쪽 하단이든 같은 방식으로 탐지되어야 한다.

### 정형 데이터에 적용하면?

정형 데이터를 CNN에 넣으려면, 피처 벡터 \([x_1, x_2, \ldots, x_n]\)를 1D 시퀀스처럼 취급해야 한다.

문제는:

- **컬럼 순서가 의미 없다** — \(x_3\)과 \(x_4\)가 인접해 있다는 것에 의미가 없음
- **필터 공유가 부적절** — "DSR과 연체횟수의 조합"을 탐지하는 필터가, "소득과 근속연수"에도 동일하게 적용되는 것이 합리적이지 않음
- **컬럼을 재배열하면 결과가 달라짐** — 이미지에서는 픽셀 순서 자체가 정보지만, 테이블에서는 컬럼 순서가 임의적이다. 의미 없는 순서에 따라 모형이 달라진다

!!! warning "요약"
    CNN의 지역성과 이동 등변성은 정형 데이터에서 **해가 되는 사전 가정**이다. Weight sharing이 오히려 다른 변수 조합마다 다른 패턴을 학습하는 것을 방해한다.

---

## 4.3 RNN/LSTM — 시계열 접근의 시도와 한계

### RNN의 핵심 구조

RNN(Recurrent Neural Network)은 **순차적 입력**을 처리하기 위한 아키텍처다. 이전 시점의 은닉 상태(hidden state)를 다음 시점에 전달하여, 과거 정보를 누적한다.

$$
\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}\right)
\tag{1}
$$

여기서 \(\mathbf{x}_t\)는 시점 \(t\)의 입력, \(\mathbf{h}_t\)는 시점 \(t\)의 은닉 상태, \(\mathbf{W}_{hh}\)가 **순환 가중치**로, 과거 정보를 현재에 전달하는 핵심 장치다. 활성함수는 보통 tanh를 쓴다.

### LSTM — Vanishing Gradient의 해법

기본 RNN은 시퀀스가 길어지면 [Vanishing Gradient](nn_basics.md) 문제로 장기 의존성을 학습하기 어렵다. LSTM(Long Short-Term Memory)은 **셀 상태(Cell State)**와 **게이트(Gate)** 메커니즘으로 이를 완화한다. LSTM은 Hochreiter & Schmidhuber(1997)[^1]가 입력·출력 게이트와 셀 상태로 처음 제안했고, 지금 표준이 된 **Forget Gate**는 Gers, Schmidhuber & Cummins(2000)[^3]가 추가했다. Forget Gate가 없으면 셀 상태가 계속 누적되어 커질 수 있는데, 이를 스스로 비우는(reset) 장치가 Forget Gate다. 아래 식은 Forget Gate가 포함된 현재의 표준형이다.

| 게이트 | 역할 |
|-------|------|
| **Forget Gate** \(\mathbf{f}_t\) | 이전 셀 상태에서 어떤 정보를 **버릴지** 결정 |
| **Input Gate** \(\mathbf{i}_t\) | 새로운 정보 중 어떤 것을 셀 상태에 **추가할지** 결정 |
| **Output Gate** \(\mathbf{o}_t\) | 셀 상태에서 어떤 정보를 **출력할지** 결정 |

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
\tag{2}
$$

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
\tag{3}
$$

셀 상태 \(\mathbf{C}_t\)는 **정보의 고속도로**처럼 작동한다. Forget Gate가 불필요한 정보를 지우고, Input Gate가 새 정보를 추가하며, 그래디언트가 이 경로를 따라 장기간 유지된다.

!!! note "ratsgo's blog"
    한국어로 LSTM을 이해하려면 이기창의 블로그 포스트 ["RNN과 LSTM"](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)이 가장 좋은 출발점이다. 게이트 구조와 역전파 흐름을 수식과 함께 직관적으로 설명한다.

!!! note "colah's blog"
    원문으로는 Christopher Olah의 ["Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)[^2]가 가장 좋은 자료다. 게이트의 흐름을 시각적으로 설명하며, 이후 GRU 등 변형 구조까지 다룬다.

---

## 4.4 신용평가에서 RNN/LSTM 시도 — 실무 경험

### 시점별 변수 구성

신용평가 데이터에서 시계열 구조를 만들 수 있는 대표적인 방법은 **시점별 동일 집계 기준 변수**를 시퀀스로 구성하는 것이다.

```
시점 t-12: 12개월 전 카드 사용금액
시점 t-11: 11개월 전 카드 사용금액
...
시점 t-1:  1개월 전 카드 사용금액
```

동일한 항목(카드 사용금액)을 12개 시점으로 나열하면, 하나의 시퀀스가 된다. 여러 항목(사용금액, 결제금액, 연체금액, ...)을 병렬로 구성하면, 각 시점에 다차원 피처 벡터를 가진 **다변량 시계열**이 된다.

### 테스트 결과: Tree 대비 이득 없음

!!! tip "저자 경험"
    시점별 변수 구성으로 LSTM 네트워크를 테스트했으나, **트리 기반 모형(XGBoost/LightGBM) 대비 유의미한 성능 이득은 없었다.**

    이 결과는 Tabular 데이터에서 DL이 트리에 밀리는 일반적 현상의 연장선이다. 시계열 구조를 부여하더라도, 12개월 정도의 집계 변수 시퀀스에서 LSTM이 포착할 만큼 복잡한 시간 의존성이 충분하지 않았던 것으로 보인다.

!!! note "비슷한 비교를 직접 해 본다면"
    LSTM과 트리를 공정하게 비교하려면 다음을 맞춰야 한다.

    - **같은 정보**: 트리에도 같은 시점별 변수(12개 시점 원값)와, 실무에서 쓰는 요약 항목(평균·추세·변동성)을 함께 넣는다. 트리에 요약 항목만 주고 LSTM에 원 시퀀스를 주면 정보량이 달라 비교가 되지 않는다.
    - **정적 변수 결합**: 소득·보유 상품 같은 시점 없는 변수는 LSTM의 마지막 은닉 상태와 이어 붙여(concatenate) 출력층에 넣는 방식이 일반적이다. 트리에는 그냥 함께 넣는다.
    - **같은 분할과 지표**: 같은 학습·검증 분할(고객 단위)에서 같은 지표(KS, AR)로, 최종적으로는 OOT에서 비교한다.
    - **튜닝 노력의 균형**: 한쪽만 공들여 튜닝하면 비교가 기운다. LSTM은 학습률·은닉 차원·드롭아웃·시퀀스 정규화에 민감하다.

### 왜 이득이 없는가 — 구조적 이유

**1. 시퀀스가 짧다**

LSTM이 강점을 발휘하려면 긴 시퀀스에서의 장기 의존성(long-range dependency)이 필요하다. 자연어 처리에서는 수백~수천 토큰의 시퀀스를 다루지만, 신용평가의 시점별 변수는 기껏해야 **12~24개 시점**이다. 이 정도 길이에서는 단순한 통계 요약(평균, 추세, 변동성)으로 대부분의 정보를 포착할 수 있다.

**2. 시점별 변수 하나하나는 의미가 약하고 불안정하다**

"3개월 전 카드 사용금액"이라는 변수는 단독으로는 해석과 운영이 까다롭다.

- 특정 월 하나의 값은 계절성(명절, 연말)이나 일시적 지출에 크게 흔들린다. 관측 시점이 고객마다 다르므로, 같은 "3개월 전"이라도 어떤 고객에게는 명절 달이고 어떤 고객에게는 평범한 달이다
- 리스크와 관련된 것은 특정 월의 값보다 **수준(평균)**, **방향(추세)**, **흔들림(변동성)**인 경우가 많다
- 시점별 변수 12개를 모두 모형에 넣으면 변수 수가 늘고, 개별 월 변수의 PSI 모니터링·설명도 번거로워진다

그래서 실무에서는 시점별 변수 대신 **집계 기간 요약 항목**을 사용하는 경우가 많다.

| 시점별 변수 (시퀀스) | 요약 항목 (스칼라) |
|---|---|
| 1개월 전, 2개월 전, ..., 12개월 전 사용금액 | 최근 6개월 평균 사용금액 |
| | 최근 12개월 누적 사용금액 |
| | 최근 3개월 vs 이전 3개월 증감률 |

요약 항목으로 가공하는 순간 시퀀스 구조는 사라지고, 일반적인 Tabular 피처가 된다. 그리고 Tabular에서는 트리가 강하다.

**3. 트리도 시간 패턴을 잡는다**

트리 모형에 시점별 변수 12개를 그냥 피처로 넣으면, 트리는 **split 구조로 시점 간 차이를 자연스럽게 포착**한다. 예를 들어:

```
IF (3개월전_사용금액 > 100만) AND (12개월전_사용금액 < 50만)
→ 최근 사용금액 급증 패턴
```

RNN이 hidden state로 누적하는 정보를, 트리는 **여러 시점 변수의 조합 split**으로 직접 잡아낸다. 12개 시점 정도의 짧은 시퀀스에서는, 이 직접적인 접근이 RNN의 순환 구조보다 오히려 효율적이다.

---

## 4.5 그래서 CNN · RNN은 언제 쓰는가

CNN과 RNN이 빛을 발하는 영역은 명확하다 — **그 아키텍처의 사전 가정이 데이터 구조와 일치할 때**다.

| 아키텍처 | 빛나는 영역 | 신용평가와의 거리 |
|---------|-----------|----------------|
| **CNN** | 이미지 분류, 객체 탐지 | 멀다 — 정형 데이터에 공간 구조 없음 |
| **RNN/LSTM** | 자연어 처리, 긴 시계열 예측 | 일부 접점 — 거래내역 시계열화 가능하나 이득 제한적 |
| **Transformer** | NLP, 음성, 최근에는 범용 | 연구 단계 — FT-Transformer 같은 Tabular용 구조, 소규모 데이터(약 1만 건 이하)에서 GBDT를 앞섰다고 보고된 사전학습형 TabPFN(Hollmann et al., *Nature*, 2025) 등이 있으나 신용평가 규모·운영 환경 검증은 제한적 |

!!! note "예외적 사례"
    금융권에서 RNN/Transformer가 검토되는 경우도 있다.

    - **통장·카드 거래내역 원본** — 건별 거래를 시퀀스로 모형화 (집계하지 않고 raw transaction 사용)
    - **이상거래탐지(FDS)** — 실시간 거래 패턴에서 이상을 탐지
    - **시장 리스크** — 주가·환율 등 고빈도 시계열 예측

    이들은 **본질적으로 시계열인 데이터**에 시계열 모형을 적용하는 것이므로 자연스럽다. 신용평가처럼 **집계된 정형 데이터**에 억지로 시계열 구조를 부여하는 것과는 다르다.

---

## 4.6 정리: 아키텍처 선택의 원칙

```
데이터 구조         →  아키텍처         →  신용평가 적합도
─────────────────────────────────────────────────────
이미지/공간 구조     →  CNN             →  ✗
긴 시퀀스/순서       →  RNN/LSTM        →  △ (이득 제한적)
정형/테이블          →  트리 앙상블      →  ◎
정형 + DL 시도       →  TabNet/MLP      →  △ (튜닝 부담 대비 이득 제한적)
```

모형 선택의 첫 번째 질문은 "어떤 알고리즘이 최신인가?"가 아니라, **"내 데이터의 구조에 어떤 사전 가정이 맞는가?"**이다.

---

## 참고 자료

[^1]: Sepp Hochreiter & Jürgen Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 1997. LSTM 원 논문. 셀 상태와 입력·출력 게이트로 장기 의존성 학습 문제를 다룸.

[^3]: Felix A. Gers, Jürgen Schmidhuber & Fred Cummins, "Learning to Forget: Continual Prediction with LSTM", *Neural Computation*, 12(10), 2000. Forget Gate를 도입한 논문.

[^2]: Christopher Olah, "Understanding LSTM Networks", *colah's blog*, 2015. [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). LSTM 구조를 가장 직관적으로 설명한 블로그 포스트.

- Stanford CS231n: Convolutional Neural Networks for Visual Recognition — CNN의 표준 교재
- Stanford CS224n: Natural Language Processing with Deep Learning — RNN/LSTM/Transformer의 NLP 적용
- Andrew Ng, *Sequence Models* (Course 5 of Deep Learning Specialization), Coursera, 2017. — RNN, LSTM, GRU, Attention의 기초
- Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", *NeurIPS Datasets and Benchmarks*, 2022.
- Hollmann et al., "Accurate predictions on small data with a tabular foundation model", *Nature*, 637, 2025. — TabPFN

!!! tip "다음 섹션"
    CNN과 RNN의 한계를 이해했다면, [트리 앙상블](../part3_tree_ensemble/index.md)에서 정형 데이터의 진정한 강자인 CART부터 XGBoost/LightGBM까지 다룬다.
