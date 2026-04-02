# 해석 가능한 ML: 1-Depth GBM과 EBM

## 4.1 두 세계의 접점

전통 스코어카드(로지스틱 회귀)와 머신러닝(트리 앙상블)은 각각의 강점이 명확하다.

| | 전통 스코어카드 (LR) | ML (XGBoost/LightGBM) |
|---|---|---|
| **해석 가능성** | 완벽 (점수표 한 장) | 제한적 (SHAP 필요) |
| **예측 성능** | 보통 | 높음 |
| **규제 수용성** | 높음 | 아직 제한적 |
| **비선형 포착** | 불가 (WoE로 간접 처리) | 자연스럽게 포착 |
| **변수 간 교호작용** | 없음 | 자동 학습 |

"전통의 해석 가능성을 유지하면서, ML의 성능을 가져올 수 없을까?"

이 질문에 대한 하나의 답이 **1-Depth GBM 스코어카드**다.

---

## 4.2 1-Depth GBM이란

Depth = 1인 트리는 **stump** — 단 하나의 변수로 한 번만 분할하는 트리다.

```
      [변수 A > 30?]
       /          \
  [Leaf: -0.3]  [Leaf: +0.5]
```

이 stump를 Boosting으로 수백~수천 개 쌓는 것이 1-Depth GBM이다.

### 핵심 특성: 교호작용이 없다

Depth = 1이면 각 트리가 **정확히 하나의 변수만** 사용한다. 따라서:

$$
F(\mathbf{x}) = F_0 + \sum_{t=1}^{T} h_t(x_{j(t)})
\tag{1}
$$

- 각 \(h_t\)는 하나의 변수 \(x_{j(t)}\)에만 의존
- 변수 간 교호작용(interaction)이 **구조적으로 불가능**
- 최종 모형은 **Generalized Additive Model (GAM)**과 동치

$$
F(\mathbf{x}) = F_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
\tag{2}
$$

각 \(f_j\)는 변수 \(j\)에 대한 비선형 함수로, 해당 변수를 사용한 모든 stump의 합이다.

!!! info "왜 이것이 해석 가능한가"
    각 변수의 효과 \(f_j(x_j)\)를 **독립적으로** 그릴 수 있다. 이것이 곧 SHAP Dependence Plot과 동일한 의미를 가진다 — 1-Depth에서는 SHAP value = 해당 변수의 partial effect이기 때문이다.

### 주의: 단조성(Monotonicity)은 자동 보장이 아니다

1-Depth GBM이 교호작용을 원천 차단한다는 점에서, "변수별 효과도 자동으로 단조적이지 않을까?"라고 오해하기 쉽다. **그렇지 않다.**

개별 stump은 leaf가 2개뿐이므로 당연히 단조적이다. 하지만 같은 변수에 대해 여러 stump이 **서로 다른 split point**에서 잘리면, 앙상블의 누적 효과는 비단조(non-monotonic)가 될 수 있다:

```
Tree  12:  연체일수 < 30 → -0.2,  ≥ 30 → +0.1
Tree  85:  연체일수 < 50 → +0.3,  ≥ 50 → -0.1
Tree 241:  연체일수 < 30 → +0.15, ≥ 30 → -0.05
```

이 stump들의 합산 결과, 연체일수의 shape function이 올라갔다 내려갔다 할 수 있다. 이것은 버그가 아니라, stump 합산이 **piecewise constant function을 자유롭게 근사**하기 때문에 나타나는 자연스러운 현상이다 — GAM의 유연성이란 곧 비단조적 곡선도 학습할 수 있다는 뜻이다.

스코어카드에서 이것이 문제가 되는 이유는 명확하다. "연체일수가 늘었는데 오히려 리스크가 낮아진다"는 구간이 존재하면, 심사 담당자에게도 감독기관에게도 설명할 수 없다.

**해법은 `monotone_constraints` 파라미터다.**

```python
params = {
    'max_depth': 1,
    'monotone_constraints': [1, -1, 0, ...],  # 변수별 방향 지정
}
```

- `+1`: 변수 값이 증가하면 예측값도 증가 (양의 단조)
- `-1`: 변수 값이 증가하면 예측값은 감소 (음의 단조)
- `0`: 제약 없음 (비단조 허용)

XGBoost, LightGBM, CatBoost 모두 이 파라미터를 지원하며, 규제 환경의 스코어카드라면 이는 선택이 아니라 **필수**다.

!!! warning "WoE 변환 입력에도 제약이 필요하다"
    WoE 변환을 거친 변수를 넣으면 이미 단조 순서로 정렬되어 있지만, XGBoost가 WoE 값의 중간에서 split하면서 비단조를 만들 수 있다. WoE 입력이라도 monotonic constraint를 거는 것이 안전하다. 다만 WoE 변수는 방향이 이미 정해져 있으므로, constraint 방향 설정이 단순해지는 장점이 있다 (전부 같은 방향).

### 이론적 근거: Friedman (2001)

이 아이디어는 새로운 것이 아니다. Friedman이 Gradient Boosting Machine을 처음 제안한 논문에서 이미 명시적으로 언급했다:

> Decision stump를 base learner로 사용하면, 결과 모형은 additive model이 된다.

즉, **1-Depth GBM = boosted stumps = GAM**이라는 등가 관계는 GBM의 탄생과 함께 시작된 것이다.

<div class="source-ref">
출처: Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." <em>Annals of Statistics</em>, 29(5):1189-1232.
<a href="https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full" target="_blank">논문 링크</a>
</div>

---

## 4.3 학술적 배경: GAM에서 EBM까지

1-Depth GBM 스코어카드를 이해하려면, GAM(Generalized Additive Model)에서 시작하는 학술적 계보를 알아야 한다.

### GAM의 출발점: Hastie & Tibshirani (1986)

**Generalized Additive Model**은 일반화 선형 모형(GLM)을 확장하여, 각 변수에 비선형(smooth) 함수를 허용하면서도 **가산성(additivity)**을 유지하는 모형이다.

$$
g(\mu) = \alpha + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
\tag{3}
$$

GLM이 \(\beta_j x_j\)라는 선형 함수만 허용하는 반면, GAM은 \(f_j(x_j)\)를 임의의 smooth 함수로 대체한다. 초기에는 spline이나 LOWESS로 \(f_j\)를 적합했다.

전통 WoE 기반 로지스틱 회귀 스코어카드도 사실 GAM의 한 형태다:

$$
\text{logit}(p) = \beta_0 + \beta_1 \cdot \text{WoE}_1(x_1) + \beta_2 \cdot \text{WoE}_2(x_2) + \cdots
$$

각 \(f_j(x_j) = \beta_j \cdot \text{WoE}_j(x_j)\)는 구간별 상수(계단 함수)다. 즉, **전통 스코어카드는 GAM의 가장 단순한 형태**이고, 1-Depth GBM은 같은 GAM 프레임워크 안에서 더 유연한 \(f_j\)를 학습하는 것이다.

<div class="source-ref">
출처: Hastie, T.J. and Tibshirani, R.J. (1986). "Generalized Additive Models." <em>Statistical Science</em>, 1(3):297-318.
<a href="https://projecteuclid.org/journals/statistical-science/volume-1/issue-3/Generalized-Additive-Models/10.1214/ss/1177013604.full" target="_blank">논문 링크</a>
</div>

### Lou-Caruana 연구 계보 (Microsoft Research, 2012--2021)

1-Depth GBM 스코어카드의 현대적 부활은 Microsoft Research의 **Rich Caruana** 그룹이 주도했다. 이 팀의 연구가 EBM(Explainable Boosting Machine)으로 이어지며, 현재 해석 가능한 ML의 표준이 되었다.

#### (1) Lou et al. (2012) --- 최적 GAM 적합법 비교

Lou, Caruana, Gehrke는 다양한 GAM 적합 방법(spline, kernel, tree 등)을 대규모로 비교하여, **shallow bagged trees + gradient boosting**이 가장 우수한 GAM 적합법임을 실증했다. 이것이 1-Depth GBM을 GAM의 핵심 엔진으로 자리잡게 한 논문이다.

<div class="source-ref">
출처: Lou, Y., Caruana, R., and Gehrke, J. (2012). "Intelligible Models for Classification and Regression." <em>KDD 2012</em>.
<a href="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf" target="_blank">PDF</a>
</div>

#### (2) Lou et al. (2013) --- GA\(^2\)M: 2-way 교호작용 추가

순수 GAM(1-Depth)은 교호작용을 포착하지 못한다. Lou et al.은 **GA\(^2\)M** (GAM + pairwise interactions)을 제안하여, 선택된 2-way 교호작용만 추가하면 블랙박스 모형과의 성능 격차를 상당 부분 줄일 수 있음을 보였다.

$$
F(\mathbf{x}) = F_0 + \sum_{j} f_j(x_j) + \sum_{j < k} f_{jk}(x_j, x_k)
\tag{4}
$$

<div class="source-ref">
출처: Lou, Y., Caruana, R., Gehrke, J., and Hooker, G. (2013). "Accurate Intelligible Models with Pairwise Interactions." <em>KDD 2013</em>.
<a href="https://dl.acm.org/doi/10.1145/2487575.2487579" target="_blank">논문 링크</a>
</div>

#### (3) Caruana et al. (2015) --- "해석 가능한 모형이 생명을 구한다"

이 논문은 해석 가능한 ML의 필요성을 가장 강력하게 보여준 사례로 널리 인용된다.

폐렴 위험 예측에 GA\(^2\)M을 적용한 결과, 모형이 **"천식 환자의 폐렴 사망 위험이 낮다"**는 패턴을 학습했다. 이는 실제로는 천식 환자가 더 적극적인 치료를 받았기 때문에 나타난 데이터 편향이었다. 블랙박스 모형은 이 위험한 패턴을 숨기지만, GAM 구조 덕분에 임상의가 즉시 식별하고 교정할 수 있었다.

!!! warning "신용평가에서도 같은 위험이 존재한다"
    예컨대, 기존 대출 거절 데이터로 학습한 모형이 "특정 직업군은 부도율이 낮다"고 학습할 수 있지만, 이는 해당 직업군이 이미 심사에서 걸러졌기 때문일 수 있다. 해석 가능한 모형만이 이런 **selection bias**를 사전에 발견할 수 있다.

<div class="source-ref">
출처: Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., and Elhadad, N. (2015). "Intelligible Models for HealthCare." <em>KDD 2015</em>.
<a href="https://dl.acm.org/doi/10.1145/2783258.2788613" target="_blank">논문 링크</a>
</div>

#### (4) Nori et al. (2019) --- InterpretML과 EBM

Microsoft Research는 이 연구 계보를 **InterpretML** 오픈소스 라이브러리로 결실시켰다. 핵심 모형이 **Explainable Boosting Machine (EBM)**이다.

EBM은 일반적인 gradient boosting과 다른 방식으로 학습한다:

| | 일반 GBM | EBM |
|---|---|---|
| **변수 선택** | 매 스텝에서 최적 변수 선택 | **Round-robin**: 변수를 순서대로 돌아가며 학습 |
| **Bagging** | 단일 또는 외부 bagging | **Inner bagging + Outer bagging** (각 25+회) |
| **결과** | 변수 간 효과가 섞일 수 있음 | 각 변수의 효과가 깔끔하게 분리됨 |
| **교호작용** | 트리 깊이에 따라 자동 | 선택적으로 2-way만 추가 (FAST 알고리즘) |

Round-robin 방식은 각 변수의 shape function \(f_j\)를 더 안정적이고 깔끔하게 만든다. 이것이 EBM이 단순 1-Depth XGBoost보다 해석 품질이 높은 이유다.

<div class="source-ref">
출처: Nori, H., Jenkins, S., Koch, P., and Caruana, R. (2019). "InterpretML: A Unified Framework for Machine Learning Interpretability." arXiv:1909.09223.
<a href="https://arxiv.org/abs/1909.09223" target="_blank">arXiv</a> ·
<a href="https://github.com/interpretml/interpret" target="_blank">GitHub</a>
</div>

---

## 4.4 전통 LR vs 1-Depth GBM vs Full GBM 비교

| 항목 | 전통 LR (WoE) | **1-Depth GBM** | Full GBM (depth=5~6) |
|------|:---:|:---:|:---:|
| **비선형 포착** | WoE 구간화로 간접 | **자동** (stump 합산) | 자동 |
| **교호작용** | 없음 | **없음** | 있음 |
| **해석 가능성** | 점수표 | **변수별 효과 곡선** | SHAP 필요 |
| **예측 성능** | 기준 | **LR < 1-Depth < Full** | 최고 |
| **변수 처리** | Classing + WoE 필수 | Raw 입력 가능 | Raw 입력 가능 |
| **규제 수용성** | 매우 높음 | **높음 (GAM 해석)** | 제한적 |

### 성능 비교의 직관

전통 LR이 성능에서 뒤지는 주요 원인 두 가지:

1. **Classing의 정보 손실**: 연속형 변수를 수동으로 5~10개 구간으로 자르면서 세밀한 패턴이 사라짐
2. **구간 내 선형 가정**: WoE 변환 후 로지스틱 회귀가 선형 결합을 하므로, 구간 내 비선형성을 포착하지 못함

1-Depth GBM은 이 두 문제를 해결한다:

- Classing 없이 원본 변수를 직접 사용
- stump들이 자동으로 최적 분할점을 찾고, 누적하여 비선형 곡선을 형성

그러나 교호작용이 없기 때문에, 변수 간 시너지를 활용하는 Full GBM보다는 성능이 낮다.

### 성능 격차의 실증적 경향

문헌에서 보고된 일반적인 경향:

- **전통 LR → 1-Depth GBM**: AUC 1~3%p 향상 (수동 구간화의 정보 손실 회복)
- **1-Depth GBM → Full GBM**: AUC 1~3%p 추가 향상 (교호작용 포착)
- **EBM (GA\(^2\)M 모드) → Full GBM**: 격차가 상당히 줄어듦 (선택적 2-way 교호작용으로 대부분 보완)

!!! note "성능 격차는 데이터에 따라 다르다"
    교호작용이 중요한 데이터(예: 소득 × 부채비율)에서는 Full GBM과의 격차가 크고, 각 변수가 독립적으로 작용하는 데이터에서는 격차가 거의 없다. Moody's Analytics의 비교 연구에서도 해석 가능한 모형이 비제약 모형 대비 성능 손실이 크지 않다고 보고했다.

<div class="source-ref">
출처: Moody's Analytics. "Automating Interpretable Machine Learning Scorecards." Whitepaper.
<a href="https://www.moodys.com/web/en/us/insights/resources/Automating-Interpretable-Scorecards.pdf" target="_blank">PDF</a>
</div>

---

## 4.5 변수별 효과 곡선 (Shape Function)

1-Depth GBM의 가장 큰 장점은 **각 변수의 효과를 곡선으로 시각화**할 수 있다는 것이다.

전통 스코어카드에서는 각 변수의 효과가 WoE 구간별 상수(계단 함수)였다:

```
연체일수:  0~30일 → WoE -0.8
          30~60일 → WoE 0.3
          60일+   → WoE 1.5
```

1-Depth GBM에서는 같은 변수의 효과가 **연속적 곡선**으로 나타난다:

```
연체일수:  0일 → -0.9
          15일 → -0.5
          30일 → 0.1
          45일 → 0.6
          60일 → 1.2
          90일 → 1.8
          (부드러운 곡선)
```

이 곡선은 해당 변수가 부도 확률에 미치는 효과를 보여주며, **심사 담당자와 규제 기관 모두에게 직관적으로 설명 가능**하다.

### SHAP과의 관계

1-Depth GBM에서는 교호작용이 없으므로:

- **SHAP value** = 해당 변수의 **shape function 값** (mean-centered)
- SHAP dependence plot = shape function plot
- SHAP interaction value = **항상 0** (교호작용 자체가 없으므로)

이것은 Full GBM과의 결정적 차이다. Full GBM에서 SHAP은 Shapley value를 근사하는 복잡한 계산이지만, 1-Depth GBM에서는 shape function을 직접 읽는 것과 같다.

---

## 4.6 Purification과 Depth-2 분해

### Purification: 효과의 정체성 보장

GA\(^2\)M처럼 2-way 교호작용을 추가하면, main effect \(f_j\)와 interaction effect \(f_{jk}\) 사이에 **효과 누출(leakage)**이 발생할 수 있다 — 모형의 예측값은 변하지 않지만, 각 항의 해석이 달라진다.

학습된 \(f_{jk}(x_j, x_k)\)가 사실은 \(x_j\)에만 의존하는 성분을 포함할 수 있다. 이렇게 되면 \(f_j\)의 shape function이 해당 변수의 실제 주효과를 정확히 반영하지 못한다.

Lengerich et al. (2020)은 **Functional ANOVA 분해**를 기반으로 purification 알고리즘을 제안했다:

1. **교호작용 항에서 주효과 성분 추출**: \(f_{jk}(x_j, x_k)\)를 \(x_k\)에 대해 주변화하여 \(x_j\)에만 의존하는 성분 분리
2. **주효과에 재흡수**: 분리된 성분을 main effect \(f_j\)에 더함
3. **교호작용 항 정화**: 남은 잔차가 순수 교호작용

purification 후 각 \(f_j\)는 순수 주효과를, 각 \(f_{jk}\)는 순수 교호작용만 포함하며, 모형의 전체 예측값은 변하지 않는다.

!!! info "EBM과 PiML에서의 purification"
    - **InterpretML (EBM)**: 학습 후 자동으로 purification을 적용하여, `explain_global()`로 보는 shape function이 정화된 주효과를 반영한다.
    - **PiML (XGB2)**: Depth-2 XGBoost를 학습한 뒤 purification을 적용하여, main effect와 interaction을 깔끔하게 분리한다.

<div class="source-ref">
출처: Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., and Caruana, R. (2020). "Purifying Interaction Effects with the Functional ANOVA." <em>AISTATS 2020</em>.
<a href="https://arxiv.org/abs/1911.04974" target="_blank">arXiv</a>
</div>

### Depth별 트레이드오프 요약

| Depth | 교호작용 | 해석 가능성 | 성능 | 해석 도구 |
|:-----:|:--------:|:----------:|:----:|:--------:|
| **1** | 없음 (GAM) | 매우 높음 | 보통 | Shape function |
| **2** | 2-way까지 | 높음 | 좋음 | Shape function + interaction heatmap |
| **3** | 3-way까지 | 보통 | 좋음~매우 좋음 | SHAP 필요 |
| **5~6** | 복잡한 고차 교호작용 | 낮음 | 매우 좋음 | SHAP 필수 |

!!! tip "Depth = 2도 좋은 선택지"
    Depth = 2이면 2-way 교호작용까지 포착하면서도, purification으로 주효과와 교호작용을 깔끔하게 분리할 수 있다. 1-Depth보다 성능이 의미 있게 올라가는 경우가 많다. PiML 라이브러리는 **XGB2** (Depth-2 XGBoost + purification)를 별도의 해석 가능 모형 유형으로 지원한다.

!!! tip "다음 페이지"
    [도구 · 구현 · 요약](depth1_tools_impl.md) --- 오픈소스 도구, 업계 사례, 구현 예시, 참고 자료를 다룬다.
