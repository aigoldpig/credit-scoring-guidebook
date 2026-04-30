---
title: "fANOVA의 두 Measure"
---

# 2. fANOVA의 두 Measure — Joint vs Product

> 같은 fANOVA 분해라도, 적분에 사용하는 측도(measure)에 따라 결과가 달라진다.

!!! warning "작성 중"
    이 페이지는 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

## 다룰 내용

### 2.0 실험에 사용된 4가지 방법론

본 부록에서 언급하는 A-1, A-2, B-1, B-2는 다음과 같다.

| 코드 | 분해 방법 | 적분 측도 (Measure) | 설명 |
|------|-----------|---------------------|------|
| **A-1** | Mobius Inversion | Joint (조건부 분포) | 관측치 기반으로 \( E[f \mid x_S] \)를 조건부 평균으로 계산 |
| **A-2** | Mobius Inversion | Product (주변분포) | leaf value에 주변 확률을 곱하여 \( E_{\text{product}}[f \mid x_S] \) 계산 |
| **B-1** | Tensor + Mass-Moving | Joint (조건부 분포) | 트리별 coefficient 텐서를 조건부 가중치로 centering |
| **B-2** | Tensor + Mass-Moving | Product (주변분포) | 트리별 coefficient 텐서를 주변분포 가중치로 centering |

- **A-2 = B-2**는 동일한 결과를 산출하며, 계산 경로만 다르다 (Product Measure에서 Möbius와 iterative centering은 동치).
- **A-1 ≠ B-1**: Joint Measure에서 Möbius inversion은 centered ANOVA 분해를 주지 않는다. B-1 (iterative centering)만이 Hooker (2007)의 generalized functional ANOVA를 올바르게 산출한다. 따라서 Joint Measure 분해에는 **B-1만 사용**한다.
- Joint measure는 SHAP의 `tree_path_dependent`에, Product measure는 `interventional`에 대응된다.

---

### A-2 = B-2가 성립하는 이유

Product measure에서 Möbius inversion(Hoeffding 분해)은 자동으로 직교성 조건(centering condition)을 만족한다.

Möbius inversion의 핵심은 inclusion-exclusion으로 효과를 분리하는 것이다:

$$
f_S(x_S) = E_S[f \mid x_S] - \sum_{T \subsetneq S} f_T(x_T)
$$

여기서 \(E_S\)는 \(S\) 밖의 변수에 대해 적분한 조건부 기대값이다.

Product measure에서 직교성 조건 \(E[f_S \mid x_T] = 0\)이 자동으로 만족되는 이유를 2변수 interaction \(f_{12}\)에 대해 확인하면:

$$
E_{\text{prod}}[f_{12} \mid x_1 = a] = E_{\text{prod}}[f \mid x_1 = a] - f_1(a) - E_{\text{prod}}[f_2(x_2) \mid x_1 = a] - \mu
$$

Product measure에서 \(P(x_2 \mid x_1) = P(x_2)\) (독립)이므로:

$$
E_{\text{prod}}[f_2(x_2) \mid x_1 = a] = E[f_2(x_2)] = 0 \quad (f_2 \text{는 이미 centered})
$$

따라서 \(E_{\text{prod}}[f_{12} \mid x_1] = 0\)이 **자동으로 성립**한다.

Tensor centering(B-2)도 같은 조건을 반복적으로 부과하므로, 둘은 같은 해에 수렴한다. Product measure의 독립성이 Möbius inversion과 iterative centering의 **동치성을 보장**하는 핵심이다.

!!! success "실증 검증"
    CB 시뮬레이션 데이터(Phase 1, depth=2, lr=0.0001)에서
    A-2와 B-2의 Main Effect % 차이: **8.13×10⁻²⁰** — 기계 정밀도 수준에서 완전 일치.

### A-1 ≠ B-1인 이유 — Joint Measure에서 Möbius의 한계

Joint measure에서는 \(P(x_2 \mid x_1) \neq P(x_2)\)이다. 이 비독립성이 Möbius inversion의 직교성을 깨뜨린다.

A-1은 Joint 조건부 기대값을 Möbius 공식에 대입하여 효과를 계산한다:

$$
f_1(x_1) = E_{\text{joint}}[f \mid x_1] - \mu, \quad
f_2(x_2) = E_{\text{joint}}[f \mid x_2] - \mu
$$

$$
f_{12}(x_1, x_2) = f(x_1, x_2) - f_1(x_1) - f_2(x_2) - \mu
$$

이 분해에서 \(f_{12}\)의 직교성을 확인하면:

$$
E_{\text{joint}}[f_{12} \mid x_1 = a]
= E_{\text{joint}}[f \mid x_1\!=\!a] - f_1(a) - E_{\text{joint}}[f_2(x_2) \mid x_1\!=\!a] - \mu
$$

여기서:

$$
E_{\text{joint}}[f_2(x_2) \mid x_1 = a]
= \sum_b f_2(b) \cdot P(x_2 = b \mid x_1 = a)
$$

**Product measure에서는** \(P(x_2 \mid x_1) = P(x_2)\)이므로 이 값이 \(E[f_2] = 0\)이 되어 직교성이 자동 성립했다.

**Joint measure에서는** \(P(x_2 \mid x_1) \neq P(x_2)\)이므로:

$$
\sum_b f_2(b) \cdot P(x_2 = b \mid x_1 = a) \neq 0 \quad \text{(일반적으로)}
$$

따라서:

$$
\boxed{E_{\text{joint}}[f_{12} \mid x_1] = -\sum_b f_2(b) \cdot P(x_2\!=\!b \mid x_1) \neq 0}
$$

**Interaction 항이 centered되지 않는다.** 이것이 A-1의 근본적 결함이다.

이 결함의 실질적 의미:

- **Main effect가 과대추정된다** — interaction에 있어야 할 분산이 main effect로 누출
- **차이의 크기는 변수 간 상관에 비례** — 변수가 독립이면 Joint = Product이므로 A-1 = B-1이 자명하게 성립
- A-1의 분해는 Hoeffding (1948)의 원래 분해와 동일하지만, 이 분해는 product measure를 전제로 설계된 것이므로 joint measure와 결합하면 직교성이 보장되지 않음

반면 B-1(`purify_tensor_joint`)은 반복적으로 **조건부 기대값을 빼는** centering을 수행한다:

$$
\text{각 변수 } j \text{에 대해}: \quad
T_u \leftarrow T_u - E_{\text{joint}}[T_u \mid x_j]
$$

이 ALS(Alternating Least Squares) 반복은 수렴 시 **모든 하위 집합에 대해 \(E_{\text{joint}}[f_S \mid x_T] = 0\)을 강제**한다. 이것이 Hooker (2007)의 generalized functional ANOVA 정의와 정확히 일치한다.

!!! danger "실증 검증 — A-1 ≠ B-1"
    CB 시뮬레이션 데이터(Phase 1, depth=2, lr=0.0001)에서:

    | 방식 | Main % | 2-way % | Cross-cov % | 소요시간 |
    |------|:---:|:---:|:---:|:---:|
    | A-1 (Möbius+Joint) | **29.93** | 0.45 | 69.62 | 276초 |
    | B-1 (Tensor+Joint) | **15.90** | 0.13 | 83.97 | 655초 |

    A-1의 Main%가 B-1의 약 **2배**로 과대추정된다.
    이 차이는 신용정보 변수 간 상관이 높기 때문에 발생한다.
    변수가 독립이면 Joint = Product이 되어 A-1 = B-1이 자명하게 성립한다.

### 결론 — 방법 선택 가이드

| 기준 | A-1 (Möbius+Joint) | B-1 (Tensor+Joint) | A-2/B-2 (Product) |
|------|:---:|:---:|:---:|
| 완전 분해 \(f = \sum f_S\) | O | O | O |
| 직교성 \(E[f_S \mid x_T] = 0\) | **X** | O | O |
| Hooker (2007) 정의 충족 | **아님** | **맞음** | 맞음 |
| main effect 정확성 | 과대추정 | 정확 | 정확 |
| 속도 | 중간 | 느림 | 빠름 |

- **Product measure 분해**: A-2 또는 B-2 (동치). A-2가 더 빠르므로 A-2 권장
- **Joint measure 분해**: **B-1만 사용**. A-1은 직교성이 깨져 사용 불가
- depth=1에서는 트리당 변수 1개이므로, 교호작용이 구조적으로 불가능하여 A-1 = B-1이 자명하게 성립

### 2.1 Product Measure (전통적 fANOVA)

- \( f_j(x_j) = \int f(x) \prod_{k \neq j} dP(x_k) - f_0 \)
- 변수 독립 가정 → 직교성 보장 → Variance Decomposition 가능
- Hoeffding (1948), Sobol (1969)의 원래 정의

### 2.2 Joint Measure (Hooker 2007)

- \( f_j(x_j) = \int f(x) \, dP(x_{-j} \mid x_j) - f_0 \)
- 실제 데이터 분포 사용 → 직교성 깨짐 → Cross-covariance 발생
- Hooker (2007) *Generalized Functional ANOVA*

### 2.3 Cross-Covariance의 불가피성

!!! example "예시 — 실제 신용평가모형에서의 분산 분해"
    저자가 과거 개발에 참여했던 신용평가모형에 4가지 방법론을 적용한 결과, 대략 아래와 같은 추세를 보였다.

| | Joint (B-1) | Product (A-2/B-2) |
|---|---|---|
| Main | ~50% | ~50% |
| 2-way | < 1% | < 1% |
| Cross-cov | ~50% | ~50% |

- 두 measure 모두 Cross-cov가 전체 분산의 약 절반을 차지했다. 그 이유:
  - Product measure 하에서 이론적 직교는 보장되지만
  - 실제 데이터 위에서 평가하면 \( \text{Cov}_{\text{data}}(f_i, f_j) \neq 0 \)
  - 데이터 상관이 존재하는 한, 어떤 measure를 쓰든 Cross-cov는 남음

### 2.4 두 Measure의 비교표

- Reconstruction: 둘 다 성립
- mu (intercept): 서로 다른 값
- 변수 중요도 순위: 유사하지만 미세 차이
- 답하는 질문이 다를 뿐, 둘 다 수학적으로 유효

### 2.5 우리 실험 — Shape Function 비교

- B-1 vs B-2의 동일 변수 shape function 비교 그림
- Joint measure: 상관 효과가 묻어 들어가 진폭이 다름

---

<div class="source-ref" markdown>
**참고**: Hooker (2007). *Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables.* JCGS 16(3):709-732
</div>
