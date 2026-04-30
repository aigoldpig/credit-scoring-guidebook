---
title: "LOWESS 기반 미니모델링과 기업 여신"
---

# 부록 C: LOWESS 기반 미니모델링 — 재무비율 변환과 기업 여신 신용평가

> Moody's RiskCalc 모델에서 활용되는 비모수적 재무비율 변환 기법의 방법론, 사상, 효과, 그리고 실무 적용 구조에 대한 기술 분석

!!! note "이 부록의 위치"
    단변량 로지스틱 회귀 섹션의 **소매 여신 관점**을 보완하여, **기업 여신(corporate lending)** 영역에서의 미니모델링 방법론을 다룬다. Moody's RiskCalc 모델 시리즈에서 공식 사용하는 LOWESS 기반 비모수적 변환 기법의 구조와 실무 적용을 정리한다.

!!! tip "소매 vs 기업 미니모델링"
    본 가이드북의 단변량 로지스틱 회귀 섹션에서 다룬 방식은 **소매 CSS**에서의 WoE 기반 방식입니다. 이 부록에서 다루는 LOWESS 기반 미니모델링은 **Moody's RiskCalc**가 기업 여신 부도 예측에 사용하는 방식으로, 접근 철학은 같지만 기법이 다릅니다.

## 이 부록에서 다루는 내용

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [RiskCalc 프로세스와 사상](riskcalc.md) | 변환 절차, 백분위 변환, 비모수적 접근의 사상(비선형성·투명성·간결성) |
| 2 | [실증적 근거와 비율별 변환](effects.md) | 5가지 효과(비선형성·정규화·이상치·한계효과·강건성), 재무비율 범주별 변환 특성 |
| 3 | [3단계 모델 아키텍처](architecture.md) | Transform → Model → Map 구조, 프로빗 모형, 최종 매핑 |
| 4 | [직관적 이해](intuition.md) | Weight & Smoothing 본질, KNN 비교, Lookup Table, Smoothing vs 실제 불량률 |
| 5 | [실무 활용과 한계](application.md) | 기업 여신 6대 활용 영역, 적용 조건, 국내 은행 접점, 한계와 고려사항 |

---

## 미니모델링의 정의와 LOWESS 통계 기초

### 미니모델링이란

"미니모델링(Mini-Modeling)"은 Moody's가 RiskCalc 모델 시리즈에서 공식적으로 사용하는 용어다. 각 재무비율을 해당 비율의 **단변량 부도확률(univariate default probability)**로 변환하는 과정을 지칭한다.

!!! quote "원문 정의"
    Moody's RiskCalc for U.S. Banks 방법론 문서에서: 각 재무비율을 해당 비율에 대응하는 단변량 부도확률로 대체하는 첫 번째 단계를, Moody's는 "미니모델링"이라 명명했다. 이 과정이 **비선형성의 상당 부분을 포착**하고, **입력값을 공통 척도로 정규화**하며, **이상치를 통제**하고, **단변량 부도 예측을 관찰함으로써 모델 내 한계효과를 모니터링**하는 데 도움을 준다고 서술하고 있다.

    <div class="source-ref">출처: Moody's RiskCalc™ Model for Privately-Held U.S. Banks, Enterprise Risk Management, July 2002</div>

이 변환에 사용되는 핵심 기법이 **LOWESS**(Locally Weighted Scatterplot Smoothing)를 포함한 국소 회귀(local regression) 및 밀도 추정(density estimation) 기법들이다. Moody's RiskCalc v3.1 기술 문서에서는 변환 추정에 다양한 국소 회귀 및 밀도 추정 기법을 사용한다고 명시하고 있다.

---

### LOWESS — Locally Weighted Scatterplot Smoothing

!!! info "정의"
    William S. Cleveland(1979)가 제안한 **비모수적 회귀 기법**. 각 데이터 포인트에 대해 인근 데이터만을 사용하여 가중 최소자승 회귀를 수행하고, 이를 전체 데이터에 걸쳐 반복 적용함으로써 **사전에 함수형태를 가정하지 않고도** 변수 간 관계를 추정한다.

    <div class="source-ref">Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." JASA 74(368): 829-836.</div>

### 알고리즘 구조

데이터 포인트 \((x_i, y_i)\)에 대해 smoothing 추정값 \(\hat{y}_i\)를 구하는 절차는 다음과 같다:

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

**3단계 — 각 고유 \(x\)값에 대해 가중 최소자승 회귀 수행**

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

### 대역폭에 따른 smoothing 결과 비교

대역폭(bandwidth) \(f\)는 LOWESS의 가장 중요한 파라미터다. 아래 그림은 동일한 산점도에 \(f\)를 다르게 적용한 결과를 보여준다:

![LOWESS 대역폭 비교](images/lowess_bandwidth_comparison.png)

- **\(f = 0.15\)** (왼쪽): 소수의 인근 점만 사용 → 노이즈까지 따라감 (과적합)
- **\(f = 0.4\)** (가운데): 적절한 범위 → 참 관계를 잘 포착
- **\(f = 0.75\)** (오른쪽): 대부분의 점을 사용 → 국소 패턴이 사라짐 (과소적합)

### 핵심 파라미터

| 파라미터 | 역할 | 영향 |
|----------|------|------|
| **\(f\) (bandwidth/span)** | 인근 데이터 비율 (\(0 < f \leq 1\)) | 작을수록 국소 패턴에 민감, 클수록 smooth |
| **Tricube weight** | 거리에 따른 가중치 감소 | 가까운 포인트에 높은 가중치 부여 |
| **Robustness iteration** | 이상치 영향 저감 | 잔차 큰 관측치의 가중치 하향 |

---

### 입력 데이터 구조 — 관측치 단위 vs 구간 요약 테이블

LOWESS를 신용평가에 적용할 때, 입력은 두 가지 형태 중 하나다.

**(a) 관측치 단위 입력** — 차주별 \((x_i, \text{TARGET}_i)\) row를 그대로 LOWESS에 투입한다. 여기서 \(\text{TARGET}_i\)는 0/1이다. 가장 단순한 방식이고, 각 row가 표본 1개를 대표하므로 별도의 표본 가중치는 필요 없다.

**(b) 구간 요약 테이블 입력** — 백분위 구간을 먼저 만들고, 구간별로 집계한 working table을 LOWESS에 투입한다.

| interval | x_center | n_obs | n_bad | bad_rate |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.01 | 1,000 | 20 | 0.020 |
| 2 | 0.03 | 950 | 28 | 0.029 |
| 3 | 0.05 | 1,100 | 45 | 0.041 |
| ... | ... | ... | ... | ... |

여기서 \(x\)값은 구간 대표값이고, \(y\)값은 구간별 실제 불량률 \(\text{bad rate} = n_{\text{bad}} / n_{\text{obs}}\)다. 기업 여신처럼 부도 데이터가 희소한 상황에서는 (b) 방식이 계산 효율과 감사 가능성 측면에서 일반적이다.

!!! warning "구간 요약 입력에서는 가중치가 두 종류 결합된다"
    구간 요약 테이블을 쓰는 순간, WLS 가중치는 **거리 가중치 하나만으로 부족**하다. 각 구간의 bad rate가 100개 관측치에서 나온 값인지 10,000개에서 나온 값인지에 따라 신뢰도가 다르기 때문이다. 따라서 두 가중치를 곱해서 최종 가중치로 쓴다:

    $$
    w_b^{\text{final}} = \underbrace{W\!\left(\frac{x_b - x_j}{h_j}\right)}_{\text{거리 가중치 (tricube)}} \times \underbrace{n_{\text{obs}, b}}_{\text{표본 가중치}}
    $$

    거리 가중치는 "이 이웃이 추정 지점에 얼마나 가까운가"를, 표본 가중치는 "이 이웃의 bad rate가 얼마나 신뢰할 만한가"를 반영한다. 관측치 단위 입력 (a)에서는 각 row가 이미 표본 1을 대표하므로 표본 가중치가 자동으로 1로 흡수되며, 거리 가중치만 명시적으로 곱하면 된다.

---

### 한 평가점에서 LOWESS가 굴러가는 모습 — 숫자로 따라가기

알고리즘이 추상적으로 느껴진다면, 평가점 한 곳에서 어떤 숫자가 들어가고 어떤 숫자가 나오는지 따라가보면 명확해진다. 외감기업 모집단의 부채비율 percentile grid에서 평가점 \(x_j = 30\)에 대한 LOWESS 계산을 step-by-step으로 살펴보자 (가상 데이터, bandwidth \(h = 12\) percentile, tricube 가중함수).

**Step 1 · 이웃 bin 선택 + 거리 정규화**

평가점 30 주변 이웃 bin들과, 각 bin의 거리를 bandwidth로 정규화한 \(u_b = (x_b - x_j)/h\):

| bin | percentile center \(x_b\) | bad_rate | \(n_{\text{obs}}\) | 거리 \(\|x_b - 30\|\) | \(u_b\) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 6 | 20 | 0.008 | 420 | 10 | 0.83 |
| 7 | 24 | 0.012 | 410 | 6 | 0.50 |
| 8 | 28 | 0.018 | 405 | 2 | 0.17 |
| 9 | 32 | 0.024 | 400 | 2 | 0.17 |
| 10 | 36 | 0.030 | 395 | 6 | 0.50 |
| 11 | 40 | 0.040 | 390 | 10 | 0.83 |

bin 5 (\(x_b = 16\), \(u = 1.17\))는 \(\|u\| > 1\)이라 tricube가 0이 되므로 자동 제외된다.

**Step 2 · Tricube 거리 가중치 \(W(u_b) = (1 - \|u_b\|^3)^3\)**

| bin | \(\|u_b\|\) | \(W(u_b)\) |
|:---:|:---:|:---:|
| 6, 11 | 0.83 | \((1 - 0.83^3)^3 = 0.428^3 \approx 0.078\) |
| 7, 10 | 0.50 | \((1 - 0.50^3)^3 = 0.875^3 \approx 0.670\) |
| 8, 9 | 0.17 | \((1 - 0.17^3)^3 \approx 0.986\) |

평가점에 가까운 bin 8·9가 0.986로 거의 1, 가장 먼 bin 6·11은 0.078로 사실상 흔적만 남는다.

**Step 3 · 표본 가중치 결합 \(w_b^{\text{final}} = W(u_b) \times n_{\text{obs}, b}\)**

| bin | \(W(u_b)\) | \(n_{\text{obs}}\) | \(w_b^{\text{final}}\) |
|:---:|:---:|:---:|:---:|
| 6 | 0.078 | 420 | 33 |
| 7 | 0.670 | 410 | 275 |
| 8 | 0.986 | 405 | 399 |
| 9 | 0.986 | 400 | 394 |
| 10 | 0.670 | 395 | 265 |
| 11 | 0.078 | 390 | 30 |

가까운 이웃에는 약 400, 먼 이웃에는 약 30 — **두 자릿수 차이**로 영향력이 차등화된다.

**Step 4 · Local 1차 WLS 적합**

\(\widehat{\text{bad rate}} = a + b \cdot (x_b - x_j)\) 형태로 weighted least squares를 풀면, 평가점이 이웃의 거의 중앙(왼쪽 3개 vs 오른쪽 3개 대칭 가까움)이라 fitted value는 가중 평균에 매우 가깝다:

$$
T(30) \approx \frac{\sum_b w_b^{\text{final}} \cdot \text{bad\_rate}_b}{\sum_b w_b^{\text{final}}} = \frac{29.4}{1{,}396} \approx 0.021
$$

**Step 5 · Cap 적용 후 lookup table 한 칸 채우기**

\([0, 1]\) 범위 안이므로 cap이 작동할 일은 없고, 그대로 **\(T(30) = 0.021\)** 이 lookup table의 percentile 30 칸에 저장된다.

이 절차를 25개 grid 평가점 모두에서 반복하면, 25개 \(T\) 값이 채워진 lookup table이 완성된다. 차주가 부채비율 50%(percentile 30)인 외감기업 A라면, 이 표에서 0.021을 읽어 다변량 Probit으로 넘긴다.

!!! tip "이 0.021이 무엇인가"
    \(T(30) = 0.021\) — 외감기업 모집단 percentile 30(즉 부채비율이 모집단 안에서 낮은 편) 차주의 **부채비율 단변량 부도확률 추정값** 2.1%다. 다른 변수(ROA, 유동비율 등)는 무시하고 부채비율 하나만 봤을 때의 추정이며, 다변량 Probit 단계에서 다른 변수들의 \(T\) 값과 결합되어 최종 EDF가 결정된다.

!!! tip "직관적 이해"
    위 수식과 절차의 직관적 의미 — bandwidth는 왜 필요한지, 회귀선은 몇 번 긋는지, 산출물은 결국 무엇인지 — 는 [4. 직관적 이해](intuition.md)에서 상세히 다룬다.

---

<div class="source-ref" markdown>
**참고 문헌**

- Falkenstein, E., Boral, A., & Carty, L. (2000). "RiskCalc for Private Companies: Moody's Default Model." Moody's Investors Service.
- Dwyer, D., Kocagil, A., & Stein, R. (2004). "The Moody's KMV EDF RiskCalc v3.1 Model." Moody's KMV.
- Moody's Analytics (2015). "RiskCalc 4.0 France." Modeling Methodology, Quantitative Research Group.
- Kocagil, A. et al. (2002). "Moody's RiskCalc Model for Privately-Held U.S. Banks." Moody's KMV.
- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." JASA 74(368): 829-836.
</div>
