---
title: "독립 가정의 한계"
---

# 4. 독립 가정의 한계 — 왜 외삽이 발생하는가

> Product measure(독립 가정)는 이론적으로 깔끔하지만, 실제 상관된 변수에서는 학습 범위 밖의 비현실적 조합을 평가하게 된다.

!!! warning "작성 중"
    이 페이지는 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

## 다룰 내용

### 4.1 Extrapolation 문제

- Product measure: \( P(X_1, X_2) = P(X_1) \cdot P(X_2) \) 가정
- 실제로는 \( P(X_1, X_2) \neq P(X_1) \cdot P(X_2) \)
- 독립 가정 시, 학습 데이터에 존재하지 않는 조합(예: 소득 300만 + 부채 3억)도 평가
- 이런 영역에서 모형의 예측은 학습 범위 밖 → 신뢰 불가

### 4.2 Hooker, Mentch, Zhou (2021) 핵심 논지

- 논문 제목: "Unrestricted Permutation Forces Extrapolation"
- Permutation Importance, PDP, fANOVA 모두 같은 문제를 공유
- "unrestricted permutation"(= product measure로 marginalize) → 본질적으로 외삽 강제
- 해결책: conditional permutation 또는 모형을 하나 더 만들어야 함

### 4.3 Dummy 공리 위반 (Observational 쪽의 문제)

- Sundararajan & Najmi (2020) ICML:
  - Observational Shapley는 **모형이 사용하지 않는 변수**에도 SHAP != 0 가능
  - 예: 모형이 x1만 사용, x2는 미사용이지만 x1과 상관 → observational에서 x2에 credit 배분
  - 이것이 **Dummy axiom 위반**
- Interventional Shapley는 Dummy를 만족하지만, 외삽 문제를 가짐

### 4.4 양쪽 모두 완벽하지 않다

| 문제 | Observational (Joint) | Interventional (Product) |
|---|---|---|
| 비현실적 조합 평가 | 없음 | 있음 (외삽) |
| 미사용 변수에 credit | 있음 (Dummy 위반) | 없음 |
| 상관 효과 분리 | 불가 (뒤섞임) | 가능 |
| 이론적 직교 | 불가 | 가능 |

- **둘 다 한계가 있으며, 목적에 따라 선택해야 한다**

---

<div class="source-ref" markdown>
**참고**

- Hooker, Mentch, Zhou (2021). *Unrestricted Permutation Forces Extrapolation.* Statistics and Computing 31:82
- Sundararajan & Najmi (2020). *The many Shapley values for model explanation.* ICML
- Janzing, Minorics, Bloebaum (2020). *Feature relevance quantification in explainable AI: A causal problem.* AISTATS
</div>
