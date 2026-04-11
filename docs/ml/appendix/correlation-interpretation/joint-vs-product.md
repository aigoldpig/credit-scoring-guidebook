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

- A-1 = B-1, A-2 = B-2는 동일한 결과를 산출하며, 계산 경로만 다르다.
- Joint measure는 SHAP의 `tree_path_dependent`에, Product measure는 `interventional`에 대응된다.

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

| | Joint (A-1/B-1) | Product (A-2/B-2) |
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
