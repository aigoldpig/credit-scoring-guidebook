---
title: "변수 상관과 모형 해석"
---

# 부록 B: 변수 상관과 모형 해석

> 변수 간 상관이 존재할 때, 모형 해석은 왜 어려워지는가?
> fANOVA와 SHAP에서 동일하게 발생하는 근본적 한계와 실무적 대응

!!! warning "작성 중"
    이 부록은 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

!!! note "이 부록의 위치"
    [부록 A: SHAP과 fANOVA](../shap-fanova/index.md)에서 다룬 분해 이론의 **실전 한계**에 해당한다.
    부록 A가 "어떻게 분해하는가"를 다뤘다면, 부록 B는 "상관된 변수가 있을 때 분해가 왜 깔끔하지 않은가"를 다룬다.

## 목차

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [두 가지 기대값 — Observational vs Interventional](observational-vs-interventional.md) | 조건부 기대값의 두 정의, "모형에 충실" vs "데이터에 충실" |
| 2 | [fANOVA의 두 Measure — Joint vs Product](joint-vs-product.md) | Joint/Product measure 수치 비교, Cross-covariance의 불가피성 |
| 3 | [SHAP의 두 모드](shap-two-modes.md) | `tree_path_dependent` vs `interventional`, Lundberg의 입장 |
| 4 | [독립 가정의 한계 — 왜 외삽이 발생하는가](independence-extrapolation.md) | Product measure의 비현실적 조합, Dummy 공리 위반 |
| 5 | [실무 권장 — 상관 변수 해석 전략](practical-guide.md) | 양쪽 결과 비교, EBM Purification, 해석 시 주의점 |

---

<div class="source-ref" markdown>
**핵심 참고 문헌**

- Chen, Lundberg, Lee (2020). [True to the Model or True to the Data?](https://arxiv.org/abs/2006.16234)
- Hooker (2007). Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables. *JCGS*
- Sundararajan & Najmi (2020). The many Shapley values for model explanation. *ICML*
- Hooker, Mentch, Zhou (2021). Unrestricted Permutation Forces Extrapolation. *Statistics and Computing*
- Lengerich et al. (2020). Purifying Interaction Effects with the Functional ANOVA. *AISTATS*
</div>
