# fANOVA 개념과 Purification

SHAP이 개별 예측을 변수별로 **분배**하는 도구라면, Functional ANOVA(fANOVA)는 모형의 예측 함수 자체를 주효과와 교호작용으로 **분리**하는 프레임워크다. 이 페이지는 fANOVA와 Purification을 **직관 수준**에서 다루고, SHAP과의 차이와 실무에서 쓸 때의 주의점을 정리한다. 수학적 정의와 알고리즘은 [부록 A: SHAP과 Functional ANOVA](../appendix/shap-fanova/index.md)에, 변수가 상관되어 있을 때의 문제는 [부록 B: 변수 상관과 모형 해석](../appendix/correlation-interpretation/index.md)에 있다.

---

## 5.1 왜 "ANOVA"인가

통계학의 분산분석(ANOVA)은 "직급이 소득에 영향을 주는가?" 같은 질문에 답하기 위해, 소득의 총 변동을 **요인별 변동**(직급 간 차이)과 **나머지**(직급 내 차이)로 쪼갠다. 요인이 둘이면 "직급 효과 + 성별 효과 + 직급×성별 교호작용 + 잔차"로 나눈다. 핵심은 **총 변동을 요인별로 쪼개고, 주효과와 교호작용을 분리하는 구조**다.

Functional ANOVA는 같은 아이디어를 데이터가 아니라 **모형의 예측 함수** \(\hat{f}\)에 적용한다.

$$
\hat{f}(\mathbf{x}) = f_0 + \sum_j f_j(x_j) + \sum_{j<k} f_{jk}(x_j, x_k) + \cdots
\tag{1}
$$

"직급이 소득 변동을 얼마나 설명하는가?" 대신 **"소득 변수가 모형 출력의 변동을 얼마나 설명하는가?"**를 묻는 것이다. 통계적 검정이 아니라 **함수 분해 도구**라는 점이 전통 ANOVA와 다르다. 이 분해는 Hoeffding(1948)이 확률론에서 처음 제시했고, Sobol(1969)이 독립적으로 재발견했으며, Hooker(2004, 2007)가 ML 모형 해석에 적용했다. 상세한 배경은 [부록 A — Functional ANOVA](../appendix/shap-fanova/functional-anova.md)에서 다룬다.

---

## 5.2 핵심 아이디어: 분배 vs 분리

SHAP과 fANOVA는 모두 "변수별 기여"를 구하지만, **교호작용을 처리하는 방식**이 근본적으로 다르다.

| | **SHAP** | **Functional ANOVA** |
|---|---|---|
| 사상(思想) | **분배**(distribute) | **분리**(decompose) |
| 교호작용 처리 | 관여한 변수들에게 **나눠 줌** | 별도의 **교호작용 항으로 분리** |
| 결과물 | 변수당 하나의 숫자 \(\phi_j\) | 주효과 \(f_j\) + 교호작용 \(f_{jk}\) + ... |
| 해석 단위 | **개별 샘플** (local) | **함수 전체** (global) |

### 구체적 예시

소득(\(X_1\))과 부채(\(X_2\))가 교호작용을 일으킨다고 하자 — 고소득+저부채일 때 부도 확률이 비선형적으로 급감하는 시너지가 있다.

**SHAP의 처리** — 교호작용의 기여분을 관여한 변수들에게 나눠준다. 변수들이 독립이고 interventional 방식으로 계산하는 이상적인 경우, 2-way 교호작용은 정확히 반씩 나뉜다:

$$
\phi_{\text{소득}} = \underbrace{f_1\text{의 기여}}_{\text{주효과}} + \underbrace{\tfrac{1}{2} f_{12}\text{의 기여}}_{\text{교호작용의 절반}}
$$

편리하지만, 대가가 있다. **소득이 동일한 두 차주라도 소득의 SHAP 값이 다를 수 있다:**

> 차주 A: 소득 5,000만 원, 부채 500만 원 → 소득의 SHAP = **-0.8**
>
> 차주 B: 소득 5,000만 원, 부채 2억 원 → 소득의 SHAP = **-0.2**

교호작용 기여분이 다른 변수의 값에 따라 달라지기 때문이다. 이것이 SHAP이 태생적으로 **local 해석**일 수밖에 없는 이유다.

**fANOVA의 처리** — 교호작용을 분배하지 않고, 별도의 항으로 분리한다:

- \(f_1(X_1)\): 소득만의 효과 — 소득이 같으면 **항상 같은 값**
- \(f_2(X_2)\): 부채만의 효과
- \(f_{12}(X_1, X_2)\): 소득과 부채의 **교호작용** — 주효과로는 설명할 수 없는 시너지

!!! warning "왜 이 차이가 중요한가 — 규제 관점"
    심사역이 "소득 변수의 효과가 단조적인가?"라고 물었을 때:

    - **SHAP**: 교호작용 기여분이 섞여 있어 특정 구간에서 비단조 패턴이 나타날 수 있고, 그것이 소득 자체의 비단조성인지 교호작용 때문인지 **구분하기 어렵다**
    - **fANOVA**: \(f_{\text{소득}}\)이 주효과만 담으므로 단조성을 **직접 확인**할 수 있다. 다만 소득과 강하게 상관된 변수가 있으면, 그 변수의 효과 일부가 소득의 주효과에 섞여 들어올 수 있다(5.3절)

---

## 5.3 분해의 기준 — 어떤 분포(측도)로 나누는가

식 (1)의 분해는 그대로는 하나로 정해지지 않는다. 같은 예측 함수를 주효과와 교호작용으로 나누는 방법은 무수히 많기 때문이다([부록 A — Non-identifiability](../appendix/shap-fanova/non-identifiability.md)). fANOVA는 "각 항은 **어떤 분포 아래에서** 평균이 0이 되어야 한다"는 조건을 걸어 분해를 하나로 정한다. 따라서 **분해 결과는 이 기준 분포(측도, measure)를 무엇으로 고르느냐에 따라 달라진다.** fANOVA에 분포 가정은 필요 없지만, 측도 선택은 반드시 필요하다.

| | **Product measure** (독립 가정) | **Joint measure** (실제 분포) |
|---|---|---|
| 기준 분포 | 변수별 주변분포의 곱 — 변수들이 독립이라고 보고 평균 | 데이터에 실제로 함께 나타나는 분포 (Hooker, 2007) |
| 각 항의 직교성 | 성립 | 변수들이 상관되어 있으면 깨짐 |
| 분산 분해 | 정확히 성립 (5.4절) | 항들 사이에 **공분산 항**이 남음 |
| 약점 | 현실에 없는 변수 조합까지 평균에 포함(외삽) | 상관된 변수끼리 효과의 경계가 흐려짐 |

트리 모형을 분해할 때 "각 구간의 학습 샘플 수를 가중치로 쓴다"는 것도 측도를 고르는 행위다. 두 변수 구간의 **결합** 샘플 수를 쓰면 joint measure에 가깝고, 변수별 **주변** 샘플 수의 곱을 쓰면 product measure다. 변수들이 서로 독립이면 두 결과가 같지만, 소득·부채·한도처럼 상관된 신용 변수에서는 shape function과 분산 비중이 달라질 수 있다. 두 측도의 비교와 실무 권장은 [부록 B — Joint vs Product](../appendix/correlation-interpretation/joint-vs-product.md)에서 다룬다.

---

## 5.4 Variance Decomposition — "교호작용이 실제로 중요한가?"

Product measure에서는 각 항이 서로 직교하므로, **모형 출력의 총 분산을 항별 분산의 합으로 정확히 나눌 수 있다:**

$$
\sigma^2(\hat{f}) = \sum_{S \subseteq \{1,\ldots,p\},\ S \neq \varnothing} \sigma^2(f_S)
\tag{2}
$$

변수들이 상관되어 있고 joint measure로 분해했다면, 식 (2)에 항들 사이의 공분산이 더해진다.

$$
\sigma^2(\hat{f}) = \sum_{S} \sigma^2(f_S) + \sum_{S \neq T} \text{Cov}(f_S, f_T)
\tag{3}
$$

이때는 각 항의 분산 비중을 더해도 100%가 되지 않으며, 공분산 몫이 클수록 "이 변수가 몇 %를 설명한다"는 해석이 흐려진다. 공분산 몫의 크기 자체가 **변수 상관이 해석을 얼마나 흐리는지** 알려주는 진단 지표가 된다.

그럼에도 이 분해가 실무적으로 중요한 이유는, **main effect와 교호작용의 상대적 비중**을 정량적으로 가늠할 수 있기 때문이다.

!!! example "신용평가 모형 예시 (product measure 기준 가상 예시)"

    | component | 분산 비중 |
    |---|---|
    | \(f_{\text{소득}}\) (main) | 22% |
    | \(f_{\text{DTI}}\) (main) | 15% |
    | \(f_{\text{연체}}\) (main) | 35% |
    | 기타 main effects | 20% |
    | \(f_{\text{소득,DTI}}\) (interaction) | 5% |
    | 기타 interactions | 3% |
    | **합계** | **100%** |

    **main effect가 전체 분산의 92%를 설명하고, 교호작용은 8%다.** 이 모형에서는 교호작용의 비중이 작으므로, 교호작용 없는 가법 모형으로도 비슷한 성능을 낼 가능성이 있다. 다만 출력 분산의 비중은 변별력(KS, AR) 기여와 같은 것이 아니므로, 실제로 가법 모형을 적합해 OOT 성능을 비교해서 확인한다.

SHAP의 `mean(|SHAP|)`로는 이 판단을 하기 어렵다. SHAP 값에는 주효과와 교호작용이 녹아들어 있어서, "우리 모형에서 교호작용이 실제로 중요한가?"라는 질문에 직접 답하지 못한다(개별 쌍의 교호작용 몫을 보려면 SHAP interaction value를 따로 계산해야 한다). fANOVA의 Variance Decomposition은 이 질문에 **숫자로** 답한다.

---

## 5.5 Purification — 효과 누출 보정

### 문제: 학습 과정에서 효과가 섞인다

GA\(^2\)M이나 depth-2 XGBoost를 학습하면, main effect \(f_j\)와 interaction \(f_{jk}\) 사이에 **효과 누출(leakage)**이 발생한다. 모형의 예측값은 변하지 않지만, 각 항의 해석이 왜곡된다. Purification(Lengerich et al., 2020)은 교호작용 항에 섞여 들어간 주효과를 찾아 주효과 항으로 되돌리는 사후 처리다.

### DTI/소득 숫자 예시

DTI와 소득, 두 변수가 각각 2구간(Low, High)인 depth-2 모형이 다음과 같이 학습되었다고 하자. 계산을 단순하게 하기 위해 **네 칸의 샘플 수가 모두 같다(균등 가중)**고 가정한다. 샘플 수가 다르면 아래의 평균이 가중평균이 된다.

**학습 결과 (purification 전)** — main effect는 전부 0이다.

| component | 값 |
|---|---|
| \(f_{\text{DTI}}\)(Low), \(f_{\text{DTI}}\)(High) | 0, 0 |
| \(f_{\text{소득}}\)(Low), \(f_{\text{소득}}\)(High) | 0, 0 |

| 교호작용 \(f_{12}\) | 소득=Low | 소득=High |
|---|---:|---:|
| **DTI=Low** | +0.1 | −0.7 |
| **DTI=High** | +0.5 | +0.1 |

해석하면 **"DTI도 소득도 개별 효과는 없고, 오직 두 변수의 조합만 중요하다."** — 정말 그럴까?

**1단계 — 소득의 누출 찾기**: 소득 구간별로 교호작용 열의 평균을 구한다.

- 소득=Low 열: \((0.1 + 0.5)/2 = +0.3\)
- 소득=High 열: \((-0.7 + 0.1)/2 = -0.3\)

이 값은 소득에만 의존하므로 **소득의 주효과**다. 교호작용에서 빼고 \(f_{\text{소득}}\)으로 옮긴다.

| 1단계 후 \(f_{12}\) | 소득=Low | 소득=High |
|---|---:|---:|
| **DTI=Low** | −0.2 | −0.4 |
| **DTI=High** | +0.2 | +0.4 |

**2단계 — DTI의 누출 찾기**: 이번에는 DTI 구간별 행 평균을 구한다. DTI=Low 행은 \(-0.3\), DTI=High 행은 \(+0.3\)이다. 이를 \(f_{\text{DTI}}\)로 옮기면 교호작용에는 다음만 남는다.

| 2단계 후 \(f_{12}\) | 소득=Low | 소득=High |
|---|---:|---:|
| **DTI=Low** | +0.1 | −0.1 |
| **DTI=High** | −0.1 | +0.1 |

**결과:**

| | Purification 전 | Purification 후 | 해석 변화 |
|---|---|---|---|
| \(f_{\text{DTI}}\) | Low=0, High=0 | Low=**−0.3**, High=**+0.3** | **"DTI 높으면 위험 ↑"** |
| \(f_{\text{소득}}\) | Low=0, High=0 | Low=**+0.3**, High=**−0.3** | **"소득 높으면 위험 ↓"** |
| 교호작용 크기 | 전체 효과 지배 | **±0.1**로 축소 | 교호작용은 작음 |

네 칸 모두 예측값이 그대로다. 예를 들어 DTI=Low·소득=High 칸은 전에는 \(0 + 0 - 0.7 = -0.7\), 후에는 \(-0.3 - 0.3 - 0.1 = -0.7\)이다. **Purification은 해석만 바꾸고 모형은 바꾸지 않는다.**

!!! note "핵심 교훈"
    학습 결과만 보면 "교호작용이 전부"였지만, purification 후에는 **main effect가 핵심이고 교호작용은 작은** 것으로 드러났다. 효과 누출이 해석을 얼마나 왜곡할 수 있는지를 보여주는 사례다. 같은 예시를 3단계 이상 반복하는 일반 알고리즘(Mass-Moving), 비균등 가중, 수렴 조건은 [부록 A — Purification](../appendix/shap-fanova/purification.md)에서 다룬다.

### 실무에서의 적용

- **InterpretML (EBM)**: purification이 기본으로 적용되지 않으므로 별도로 적용해야 한다([EBM](ebm.md) 4.6절)
- **PiML (XGB2)**: depth-2 XGBoost에 purification을 결합한 모형을 내장 지원
- 논문 저자들이 공개한 [GAM Purification](https://github.com/AdaptInfer/gam_purification) 라이브러리로도 적용 가능

어느 도구든 위 예시의 "평균"을 **어떤 가중치로 구하느냐**(5.3절의 측도)를 확인해야 한다. 같은 모형이라도 가중치에 따라 purification 결과가 달라진다.

<div class="source-ref">
출처: Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., and Caruana, R. (2020). "Purifying Interaction Effects with the Functional ANOVA." <em>AISTATS 2020</em>.
<a href="https://arxiv.org/abs/1911.04974" target="_blank">arXiv</a> ·
Hooker, G. (2007). "Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables." <em>Journal of Computational and Graphical Statistics</em>, 16(3).
</div>

---

## 5.6 Model Complexity ↔ Interpretability Trade-off

fANOVA는 이론적으로 **아무리 복잡한 트리 모형이라도** 주효과와 교호작용으로 분해할 수 있다. 그러나 분해할 수 있다는 것과, 그 결과를 사람이 이해할 수 있다는 것은 **별개의 문제**다.

예를 들어 변수 30개로 학습한 depth-8 XGBoost 500 라운드 모형을 fANOVA로 분해하면:

- main effect의 shape function이 수십 개의 bin으로 잘게 쪼개져, 곡선이 아니라 **노이즈에 가까운 지그재그**가 된다
- 2-way 교호작용만 해도 \(\binom{30}{2} = 435\)개 조합이 나오고, 각각이 복잡한 heatmap이 된다. 여기에 3-way부터 8-way까지의 항이 더 있다
- 심사역이나 검증 담당자에게 "이 435개 heatmap을 보시면 모형이 이해됩니다"라고 할 수는 없다

결국 fANOVA가 실무에서 **직관적 해석 도구로 기능**하려면, 모형 자체의 복잡도가 일정 수준 이내여야 한다.

### 복잡한 모형에서 fANOVA가 실제로 전달할 수 있는 것

성능 중심으로 개발된 깊은 트리 모형이라 하더라도, fANOVA를 통해 실무적으로 의미 있는 해석을 끌어낼 수 있는 영역은 두 가지다:

**(1) Main effect의 shape function** — 변수 하나의 효과는 2D 곡선이므로, bin이 다소 많더라도 **전체적인 방향성과 단조성**은 확인할 수 있다. "소득이 올라가면 부도 확률이 내려간다"는 수준의 직관은 복잡한 모형에서도 전달 가능하다.

**(2) 분산 기여도가 큰 상위 교호작용** — Variance Decomposition(5.4절)으로 교호작용의 분산 비중을 정렬한 뒤, **상위 2~3개 조합**만 보고한다. 다만 각 변수의 구간이 지나치게 세분화되어 있으면 heatmap이 읽히지 않으므로, 운영 평점표로 옮길 때는 관행([해석 가능성](interpretability.md) 참고)에 맞춰 변수당 5개 안팎 구간으로 묶어 요약할 수 있다.

!!! warning "해석 가능 모형과의 차이"
    처음부터 해석을 염두에 두고 **구간 수·교호작용 차수·방향을 제약해 설계한 모형**은 main effect와 교호작용이 적은 bin, 적은 조합으로 구성된다. 깊이만 얕게 했거나 EBM을 기본 설정으로 쓴 것만으로는 bin 수가 저절로 적어지지 않는다는 점에 주의한다([EBM](ebm.md) 4.5절). 반면 깊은 트리 모형에 fANOVA를 사후 적용하는 것은 "이미 복잡해진 결과물에서 읽을 수 있는 부분만 추려내는" 작업이다. 같은 fANOVA라도 **모형 설계 단계에서 복잡도를 제어했느냐**에 따라 해석의 품질이 근본적으로 달라진다.

!!! abstract "수학적 상세 → 부록"
    fANOVA 직교 분해의 수학적 제약 조건, Non-identifiability 문제, Mass-Moving 알고리즘의 수렴 정리, SHAP과 fANOVA의 심층 비교 등은 [부록 A: SHAP과 Functional ANOVA](../appendix/shap-fanova/index.md)에서, 상관 변수에서의 측도 선택은 [부록 B: 변수 상관과 모형 해석](../appendix/correlation-interpretation/index.md)에서 다룬다.

!!! tip "다음 페이지"
    [도구 · 구현 · 요약](depth1_tools_impl.md) --- 오픈소스 도구, 실무 파라미터, 적용 시나리오, 참고 자료를 다룬다.
