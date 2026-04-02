# CNN · RNN — 정형 데이터에서의 한계

> CNN과 RNN은 각각 이미지와 시계열에 특화된 아키텍처다. 신용평가 정형 데이터에는 맞지 않지만, 왜 안 맞는지를 이해하면 딥러닝의 **사전 가정(Inductive Bias)**이 보인다.

!!! quote "이 장의 핵심"
    모든 신경망 아키텍처에는 **"이 데이터는 이런 구조를 가지고 있을 것이다"**라는 가정이 내장되어 있다. CNN은 공간적 지역성, RNN은 시간적 순서를 가정한다. 정형 데이터는 이 두 가정 모두에 해당하지 않는다.

---

## 4.1 사전 가정 (Inductive Bias)

딥러닝 아키텍처를 이해하는 열쇠는 **사전 가정**이다.

| 아키텍처 | 사전 가정 | 가정 |
|---------|-----------|------|
| **MLP** | 없음 (범용) | 입력 간 관계에 대한 사전 가정 없음 |
| **CNN** | **공간적 지역성** + **이동 불변성** | 가까운 픽셀이 먼 픽셀보다 관련이 깊고, 패턴은 위치에 무관 |
| **RNN** | **시간적 순서** + **순차적 의존성** | 과거 정보가 현재에 영향을 주고, 순서가 중요 |
| **Transformer** | **Attention** | 모든 위치 간 관계를 학습, 순서는 Position Encoding으로 주입 |

정형(Tabular) 데이터의 특성:

- **컬럼 간 순서가 없다** — "부채비율"이 "연체횟수" 옆에 있든 멀리 있든 의미가 같다
- **공간적 지역성이 없다** — 인접 컬럼끼리 특별히 관련이 깊지 않다
- **시계열 구조가 아니다** — 각 행(고객)은 독립적인 스냅샷

!!! info "왜 MLP도 Tabular에서 트리에 밀리나?"
    MLP는 사전 가정이 없으므로 Tabular에 "맞지 않는" 것은 아니다. 문제는 **사전 가정이 부족한 것**이다. 트리는 "축 정렬 분할(axis-aligned split)"이라는 Tabular에 딱 맞는 사전 가정을 가지고 있어, 더 적은 데이터로 효율적으로 학습한다. 이 내용은 [트리 앙상블](../part3_tree_ensemble/index.md) 장에서 본격적으로 다룬다.

---

## 4.2 CNN — 왜 정형 데이터에 안 맞는가

CNN(Convolutional Neural Network)은 **합성곱 필터**가 입력의 지역적 패턴을 탐지하고, 이 필터를 전체 입력에 걸쳐 공유(weight sharing)하는 구조다.

### CNN의 핵심 가정

1. **지역성(Locality):** 의미 있는 패턴은 인접한 요소들의 조합에서 나타남
2. **이동 불변성(Translation Invariance):** 같은 패턴은 어디에 나타나든 동일하게 탐지되어야 함

이미지에서 이 가정은 완벽하게 들어맞는다. 고양이 귀의 패턴은 이미지 왼쪽 상단이든 오른쪽 하단이든 동일하게 탐지되어야 한다.

### 정형 데이터에 적용하면?

정형 데이터를 CNN에 넣으려면, 피처 벡터 \([x_1, x_2, \ldots, x_n]\)를 1D 시퀀스처럼 취급해야 한다.

문제는:

- **컬럼 순서가 의미 없다** — \(x_3\)과 \(x_4\)가 인접해 있다는 것에 의미가 없음
- **필터 공유가 부적절** — "부채비율과 연체횟수의 조합"을 탐지하는 필터가, "소득과 근속연수"에도 동일하게 적용되는 것이 합리적이지 않음
- **컬럼을 재배열하면 결과가 달라짐** — 이미지에서는 픽셀 순서를 바꾸면 안 되지만, 테이블에서는 컬럼 순서가 임의적

!!! warning "요약"
    CNN의 지역성과 이동 불변성은 정형 데이터에서 **해가 되는 사전 가정**이다. Weight sharing이 오히려 다른 변수 조합마다 다른 패턴을 학습하는 것을 방해한다.

---

## 4.3 RNN/LSTM — 시계열 접근의 시도와 한계

### RNN의 핵심 구조

RNN(Recurrent Neural Network)은 **순차적 입력**을 처리하기 위한 아키텍처다. 이전 시점의 은닉 상태(hidden state)를 다음 시점에 전달하여, 과거 정보를 누적한다.

$$
\mathbf{h}_t = \sigma\!\left(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}\right)
\tag{1}
$$

여기서 \(\mathbf{x}_t\)는 시점 \(t\)의 입력, \(\mathbf{h}_t\)는 시점 \(t\)의 은닉 상태, \(\mathbf{W}_{hh}\)가 **순환 가중치** — 과거 정보를 현재에 전달하는 핵심 장치다.

### LSTM — Vanishing Gradient의 해법

기본 RNN은 시퀀스가 길어지면 [Vanishing Gradient](nn_basics.md) 문제로 장기 의존성을 학습하기 어렵다. LSTM(Long Short-Term Memory, Hochreiter & Schmidhuber, 1997)[^1]은 **셀 상태(Cell State)**와 **게이트(Gate)** 메커니즘으로 이를 해결한다.

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

### 왜 이득이 없는가 — 구조적 이유

**1. 시퀀스가 짧다**

LSTM이 강점을 발휘하려면 긴 시퀀스에서의 장기 의존성(long-range dependency)이 필요하다. 자연어 처리에서는 수백~수천 토큰의 시퀀스를 다루지만, 신용평가의 시점별 변수는 기껏해야 **12~24개 시점**이다. 이 정도 길이에서는 단순한 통계 요약(평균, 추세, 변동성)으로 대부분의 정보를 포착할 수 있다.

**2. 시점별 변수의 해석 모호함**

"최근 3개월 전 카드 사용금액"이라는 변수는 해석이 모호하다.

- 3개월 전이 **절대적으로** 의미 있는 시점인가? (예: 특정 경제 이벤트?)
- 아니면 **상대적으로** 최근 추세를 반영하는 것인가?
- 고객마다 신청 시점이 다르므로, "3개월 전"의 의미도 달라진다

이 모호함 때문에 실무에서는 시점별 변수 대신 **집계 기간 요약 항목**을 사용하는 경우가 많다.

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
| **Transformer** | NLP, 음성, 최근에는 범용 | 연구 단계 — FT-Transformer 등 Tabular 시도 존재 |

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
정형 + DL 시도       →  TabNet/MLP      →  ○ (트리에 준하는 수준)
```

모형 선택의 첫 번째 질문은 "어떤 알고리즘이 최신인가?"가 아니라, **"내 데이터의 구조에 어떤 사전 가정이 맞는가?"**이다.

---

## 참고 자료

[^1]: Sepp Hochreiter & Jürgen Schmidhuber, "Long Short-Term Memory", *Neural Computation*, 1997. LSTM 원 논문. Vanishing Gradient 문제를 게이트 메커니즘으로 해결.

[^2]: Christopher Olah, "Understanding LSTM Networks", *colah's blog*, 2015. [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). LSTM 구조를 가장 직관적으로 설명한 블로그 포스트.

- Stanford CS231n: Convolutional Neural Networks for Visual Recognition — CNN의 표준 교재
- Stanford CS224n: Natural Language Processing with Deep Learning — RNN/LSTM/Transformer의 NLP 적용
- Andrew Ng, *Sequence Models* (Course 5 of Deep Learning Specialization), Coursera, 2017. — RNN, LSTM, GRU, Attention의 기초

!!! tip "다음 섹션"
    CNN과 RNN의 한계를 이해했다면, [트리 앙상블](../part3_tree_ensemble/index.md)에서 정형 데이터의 진정한 강자인 CART부터 XGBoost/LightGBM까지 다룬다.
