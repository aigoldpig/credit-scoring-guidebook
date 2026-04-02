# 미니모델링의 정의와 LOWESS 통계 기초

## 1.1 미니모델링이란

"미니모델링(Mini-Modeling)"은 Moody's가 RiskCalc 모델 시리즈에서 공식적으로 사용하는 용어다. 각 재무비율을 해당 비율의 **단변량 부도확률(univariate default probability)**로 변환하는 과정을 지칭한다.

!!! quote "원문 정의"
    Moody's RiskCalc for U.S. Banks 방법론 문서에서: 각 재무비율을 해당 비율에 대응하는 단변량 부도확률로 대체하는 첫 번째 단계를, Moody's는 "미니모델링"이라 명명했다. 이 과정이 **비선형성의 상당 부분을 포착**하고, **입력값을 공통 척도로 정규화**하며, **이상치를 통제**하고, **단변량 부도 예측을 관찰함으로써 모델 내 한계효과를 모니터링**하는 데 도움을 준다고 서술하고 있다.

    <div class="source-ref">출처: Moody's RiskCalc™ Model for Privately-Held U.S. Banks, Enterprise Risk Management, July 2002</div>

이 변환에 사용되는 핵심 기법이 **LOWESS**(Locally Weighted Scatterplot Smoothing)를 포함한 국소 회귀(local regression) 및 밀도 추정(density estimation) 기법들이다. Moody's RiskCalc v3.1 기술 문서에서는 변환 추정에 다양한 국소 회귀 및 밀도 추정 기법을 사용한다고 명시하고 있다.

---

## 1.2 LOWESS — Locally Weighted Scatterplot Smoothing

!!! info "정의"
    William S. Cleveland(1979)가 제안한 **비모수적 회귀 기법**. 각 데이터 포인트에 대해 인근 데이터만을 사용하여 가중 최소자승 회귀를 수행하고, 이를 전체 데이터에 걸쳐 반복 적용함으로써 **사전에 함수형태를 가정하지 않고도** 변수 간 관계를 추정한다.

    <div class="source-ref">Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." JASA 74(368): 829-836.</div>

### 알고리즘 구조

데이터 포인트 \((x_i, y_i)\)에 대해 평활값 \(\hat{y}_i\)를 구하는 절차는 다음과 같다:

**1단계 — 대역폭(bandwidth) 결정**

전체 \(N\)개 중 \(f \times N\)개의 인근 포인트를 선택한다:

$$
k = \left\lfloor \frac{N \times f - 0.5}{2} \right\rfloor
$$

**2단계 — 삼차 가중함수(tricube weight function) 적용**

![Tricube 가중함수](images/tricube_weight.png)

$$
W(u) = \begin{cases} (1 - |u|^3)^3 & \text{if } |u| < 1 \\ 0 & \text{otherwise} \end{cases} \tag{B.1}
$$

**3단계 — 각 \(x_i\)에 대해 가중 최소자승 회귀 수행**

$$
\hat{y}_i = \text{WLS}\left(x_j,\; y_j,\; W\!\left(\frac{x_j - x_i}{h_i}\right)\right) \tag{B.2}
$$

**4단계 — (선택) 강건성(robustness) 반복**

잔차 기반 bisquare 가중치를 재적용한다:

$$
r_i = y_i - \hat{y}_i
$$

$$
B(u) = (1 - u^2)^2 \quad \text{if } |u| < 1 \tag{B.3}
$$

### 대역폭에 따른 평활 결과 비교

대역폭(bandwidth) \(f\)는 LOWESS의 가장 중요한 파라미터다. 아래 그림은 동일한 산점도에 \(f\)를 다르게 적용한 결과를 보여준다:

![LOWESS 대역폭 비교](images/lowess_bandwidth_comparison.png)

- **\(f = 0.15\)** (왼쪽): 소수의 인근 점만 사용 → 노이즈까지 따라감 (과적합)
- **\(f = 0.4\)** (가운데): 적절한 범위 → 참 관계를 잘 포착
- **\(f = 0.75\)** (오른쪽): 대부분의 점을 사용 → 국소 패턴이 사라짐 (과소적합)

### 핵심 파라미터

| 파라미터 | 역할 | 영향 |
|----------|------|------|
| **\(f\) (bandwidth/span)** | 인근 데이터 비율 (\(0 < f \leq 1\)) | 작을수록 국소 패턴에 민감, 클수록 평활 |
| **Tricube weight** | 거리에 따른 가중치 감소 | 가까운 포인트에 높은 가중치 부여 |
| **Robustness iteration** | 이상치 영향 저감 | 잔차 큰 관측치의 가중치 하향 |

!!! tip "직관적 이해"
    LOWESS는 "각 점마다 주변 이웃에게 물어본다"는 비유로 이해할 수 있다. 내 주변에 가까운 데이터일수록 큰 목소리를 내고(가중치 ↑), 먼 데이터는 작은 목소리를 낸다(가중치 ↓). 이를 전체 데이터에 걸쳐 반복하면, 사전에 "직선이다" 또는 "2차곡선이다" 같은 가정 없이도 데이터가 스스로 말하는 관계를 추출할 수 있다.
