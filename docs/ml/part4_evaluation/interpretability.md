# 해석 가능성 (Interpretability)

!!! quote "저자 노트"
    해석 가능성(Interpretable ML)은 그 자체로 하나의 분야다. 이 페이지에서는 신용평가 실무에 필요한 **핵심 개념과 도구**만 간결하게 다룬다. 더 깊은 이해가 필요하다면 Christoph Molnar의 **[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)**을 강력히 추천한다. 저자 본인도 이 책으로 많은 공부를 했으며, ML 해석 가능성에 관한 가장 체계적이고 접근하기 쉬운 자료라고 생각한다.

---

## 1.1 XAI란 무엇인가

**XAI(eXplainable Artificial Intelligence)**는 AI 모형의 의사결정 과정을 사람이 이해할 수 있도록 만드는 기술과 방법론의 총칭이다. 블랙박스 모형이 "무엇을" 예측했는지뿐 아니라, **"왜" 그렇게 예측했는지**를 설명하는 것이 핵심이다.

### 용어 정리

실무에서 혼용되지만, 학술적으로는 구분되는 개념들이 있다.

| 용어 | 영문 | 의미 |
|------|------|------|
| **해석 가능성** | Interpretability | 모형의 작동 원리를 사람이 직접 이해할 수 있는 정도. 로지스틱 회귀, 의사결정나무 등 **모형 자체가 투명**한 경우 |
| **설명 가능성** | Explainability | 블랙박스 모형에 **사후적으로 설명을 부여**하는 것. SHAP, LIME 등 별도의 설명 도구를 사용 |
| **투명성** | Transparency | 모형의 학습 과정, 데이터, 의사결정 로직이 공개되어 있는 정도 |
| **공정성** | Fairness | 모형이 특정 집단(성별, 인종 등)에 대해 체계적 차별을 하지 않는 성질 |

!!! note "Interpretability vs Explainability"
    **해석 가능한(Interpretable)** 모형은 별도 도구 없이 구조 자체로 이해된다 — 스코어카드가 대표적이다.
    **설명 가능한(Explainable)** 모형은 그 자체로는 이해하기 어렵지만, SHAP 등 외부 도구로 사후 설명이 가능하다.
    XAI는 이 두 가지를 모두 포괄하는 상위 개념이다.

### 모형 복잡도와 해석 가능성의 트레이드오프

```
해석 가능성  높음 ◀━━━━━━━━━━━━━━━━━━━━━━━━━━▶ 낮음
                │          │            │           │
            선형 회귀    의사결정나무    Random     Deep
            스코어카드                  Forest     Learning
                │          │            │           │
예측 성능    낮음 ◀━━━━━━━━━━━━━━━━━━━━━━━━━━▶ 높음
```

전통 스코어카드(로지스틱 회귀)는 **Interpretable by design**이다. 반면 트리 앙상블이나 딥러닝은 성능은 높지만 자체 해석이 어려워, **XAI 기법이 필수**가 된다.

---

## 1.2 왜 해석이 필요한가

전통 로지스틱 회귀 스코어카드는 태생적으로 해석이 쉽다. 각 변수의 WoE × β가 곧 점수이고, 점수표 한 장으로 "왜 이 고객이 이 등급인지" 설명할 수 있다.

트리 앙상블(XGBoost, LightGBM)은 수백~수천 개의 트리를 합산한 결과이므로, **모형 내부를 직접 들여다보는 것이 불가능**하다. 그러나 해석의 필요성은 사라지지 않는다.

| 이해관계자 | 요구 |
|-----------|------|
| **규제 기관** | "이 모형이 차주를 차별하지 않는다"는 근거 |
| **심사 담당자** | "왜 이 고객이 거절되었는가"에 대한 사유 |
| **모형 개발자** | 모형이 합리적인 패턴을 학습했는지 검증 |
| **경영진** | 모형 도입의 근거와 리스크 |

---

## 1.3 Global vs Local 해석

해석 가능성은 두 가지 차원으로 나뉜다.

| | **Global** | **Local** |
|---|---|---|
| **질문** | "이 모형은 전체적으로 어떤 변수를 중요하게 보는가?" | "이 특정 고객의 예측에 어떤 변수가 영향을 미쳤는가?" |
| **대상** | 모형 전체 | 개별 예측 건 |
| **용도** | 모형 검증, 변수 중요도 | 거절 사유, 심사 설명 |
| **도구** | Feature Importance, PDP, SHAP Summary | SHAP Waterfall, LIME |

---

## 1.4 PDP (Partial Dependence Plot)

**특정 변수가 예측에 미치는 평균적 영향**을 시각화하는 Global 해석 도구다.

변수 \(x_s\)의 Partial Dependence:

$$
\hat{f}_s(x_s) = \frac{1}{N} \sum_{i=1}^{N} \hat{f}(x_s, x_{C}^{(i)})
\tag{1}
$$

- \(x_s\): 관심 변수 (고정할 값을 바꿔가며)
- \(x_C^{(i)}\): 나머지 변수 (데이터의 실제 값 그대로)
- 모든 샘플에 대해 관심 변수만 바꾸고, 예측값의 평균을 구함

!!! example "신용평가 예시"
    "부채비율(DTI)"의 PDP를 그리면, DTI가 증가함에 따라 예측 부도 확률이 어떻게 변하는지 평균적 경향을 볼 수 있다. DTI 40%까지는 완만하게 증가하다가 60%를 넘으면 급격히 상승하는 패턴이 보인다면, 모형이 합리적인 관계를 학습했다고 판단할 수 있다.

<figure markdown="span">
  ![PDP — 부채비율(DTI)이 부도 확률에 미치는 평균적 영향](images/pdp_dti.png){ width="720" }
  <figcaption>DTI 40%까지는 완만하게 증가하다가 60%를 넘으면 급격히 상승하는 전형적인 PDP 패턴. 파란 음영은 95% 신뢰 구간, 하단 rug plot은 DTI 분포를 나타낸다.</figcaption>
</figure>

### PDP의 한계

- **변수 간 상호작용을 무시**: 다른 변수를 고정한 상태의 평균이므로, 교호작용 효과가 상쇄됨
- **상관된 변수 문제**: DTI와 소득이 강하게 상관되어 있을 때, DTI를 극단값으로 설정하면서 소득은 원본 그대로 유지하는 비현실적 조합이 발생

---

## 1.5 SHAP (SHapley Additive exPlanations)

[Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)가 제안한 SHAP은 현재 **ML 해석의 사실상 표준**이다. 게임이론의 Shapley Value를 ML에 적용한 것이다.

### Shapley Value의 직관

> "각 변수가 예측에 기여한 공정한 몫은 얼마인가?"

5명이 팀 프로젝트를 했는데 최종 점수가 90점이다. 각자의 기여도를 공정하게 분배하고 싶다면?

Shapley Value의 방법: **모든 가능한 팀 조합에서, 해당 멤버가 참여할 때와 참여하지 않을 때의 점수 차이를 평균**낸다.

변수 \(j\)의 SHAP value:

$$
\phi_j = \sum_{S \subseteq \{1,...,p\} \setminus \{j\}} \frac{|S|!(p-|S|-1)!}{p!} \left[\hat{f}(S \cup \{j\}) - \hat{f}(S)\right]
\tag{2}
$$

### SHAP의 핵심 성질

| 성질 | 의미 |
|------|------|
| **Efficiency** (효율성) | 모든 변수의 SHAP 합 = 예측값 - 기대값 |
| **Symmetry** (대칭성) | 기여가 동일한 변수는 같은 SHAP 값 |
| **Dummy** (무관 변수) | 예측에 영향 없는 변수의 SHAP = 0 |
| **Additivity** (가산성) | 앙상블의 SHAP = 개별 모형 SHAP의 평균 |

Efficiency 성질이 특히 중요하다:

$$
\hat{f}(x) = E[\hat{f}(X)] + \sum_{j=1}^{p} \phi_j(x)
\tag{3}
$$

개별 예측값이 **기대값 + 각 변수의 기여분**으로 정확히 분해된다.

### TreeSHAP

Shapley Value의 정확한 계산은 \(2^p\)개 조합을 평가해야 하므로 변수가 많으면 불가능하다. **TreeSHAP** ([Lundberg et al., 2020](https://www.nature.com/articles/s42256-019-0138-9))은 트리 구조를 활용하여 \(O(TLD^2)\) 시간에 정확한 SHAP을 계산한다.

- \(T\): 트리 수, \(L\): 평균 Leaf 수, \(D\): 트리 깊이
- XGBoost, LightGBM에 내장되어 있어 별도 근사 없이 정확한 값 계산 가능

---

## 1.6 SHAP 시각화

### SHAP Summary Plot (Global)

모든 샘플에 대한 SHAP 값을 한 그림에 표시. 변수 중요도와 방향성을 동시에 파악.

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
shap.summary_plot(shap_values, X_valid)
```

- x축: SHAP value (양수 = 부도 확률 ↑, 음수 = 부도 확률 ↓)
- y축: 변수 (중요도 순 정렬)
- 색상: 변수의 실제 값 (빨강 = 높음, 파랑 = 낮음)

### SHAP Waterfall Plot (Local)

**개별 고객** 한 명의 예측을 분해하여 각 변수의 기여를 보여준다.

```python
shap.plots.waterfall(shap_values[0])
```

- 기대값(base value)에서 시작하여, 각 변수가 예측을 위/아래로 밀어내는 과정을 시각화
- "이 고객이 왜 거절되었는가?"에 대한 직접적 답변

### SHAP Dependence Plot (Global)

PDP와 유사하지만, 개별 샘플의 SHAP 값을 산점도로 표시. 교호작용까지 색상으로 표현 가능.

```python
shap.dependence_plot("DTI", shap_values, X_valid)
```

!!! tip "다음 페이지"
    [SHAP — 해석의 여정](shap_in_practice.md) --- SHAP을 실무에 적용하려 했을 때 부딪히는 현실적인 벽과, 그 벽을 넘기 위한 고민을 정리한다.
