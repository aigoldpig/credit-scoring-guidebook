---
title: "변수 상관과 모형 해석"
---

# 부록 B: 변수 상관과 모형 해석

> 변수 간 상관이 존재할 때, 모형 해석은 왜 어려워지는가?
> fANOVA와 SHAP에서 동일하게 발생하는 근본적 한계와 실무적 대응

!!! note "이 부록의 위치"
    [부록 A: SHAP과 fANOVA](../shap-fanova/index.md)에서 다룬 분해 이론의 **실전 한계**에 해당한다.
    부록 A가 주로 변수 독립(product measure)을 전제로 "어떻게 분해하는가"를 다뤘다면, 부록 B는 "상관된 변수가 있을 때 분해가 왜 하나로 정해지지 않는가, 그때 무엇을 골라야 하는가"를 다룬다.

부록 B의 모든 페이지는 같은 작은 가상 예시 — 서로 상관된 두 이진 변수(현금서비스 이용 여부, 카드론 이용 여부)와 depth-2 트리 모형 — 를 공유한다. 예시는 [1절](observational-vs-interventional.md)에서 정의하고, 2절과 4절은 목적에 맞게 분포를 바꾼 변형 예시를 쓴다. 모든 수치는 각 페이지의 분포표에서 손으로 다시 계산할 수 있는 가상 수치다.

## 목차

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [두 가지 기대값 — Observational vs Interventional](observational-vs-interventional.md) | 조건부 기대값과 개입적 기대값의 정의, "모형에 충실" vs "데이터에 충실", 공통 수치 예시 |
| 2 | [fANOVA의 두 Measure — Joint vs Product](joint-vs-product.md) | Product/Joint measure의 정의와 성질, 조건부 평균 대입의 함정, 세 가지 분해 수치 비교, Cross-covariance의 불가피성 |
| 3 | [SHAP의 두 모드](shap-two-modes.md) | interventional · observational · path-dependent 가치함수, 같은 고객의 SHAP 값 비교, fANOVA와의 정확한 대응 관계 |
| 4 | [독립 가정의 한계 — 왜 외삽이 발생하는가](independence-extrapolation.md) | 외삽 문제의 수치 예시(PDP vs M-plot vs ALE), Hooker 외(2021), ALE, Dummy 공리 위반 |
| 5 | [실무 권장 — 상관 변수 해석 전략](practical-guide.md) | 질문 정하기, 양쪽 결과 비교, Cross-cov 진단, EBM·Purification, 단조 제약, 신용평가 실무 권장 |

---

<div class="source-ref" markdown>
**핵심 참고 문헌**

- Chen, Janizek, Lundberg, Lee (2020). [True to the Model or True to the Data?](https://arxiv.org/abs/2006.16234) arXiv:2006.16234
- Janzing, Minorics, Blöbaum (2020). [Feature relevance quantification in explainable AI: A causal problem](https://proceedings.mlr.press/v108/janzing20a.html). AISTATS
- Hooker (2007). [Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables](https://doi.org/10.1198/106186007X237892). *JCGS* 16(3)
- Sundararajan, Najmi (2020). [The Many Shapley Values for Model Explanation](https://proceedings.mlr.press/v119/sundararajan20b.html). ICML
- Hooker, Mentch, Zhou (2021). [Unrestricted permutation forces extrapolation](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing* 31, 82
- Apley, Zhu (2020). [Visualizing the effects of predictor variables in black box supervised learning models](https://doi.org/10.1111/rssb.12377). *JRSS-B* 82(4)
- Lengerich, Tan, Chang, Hooker, Caruana (2020). [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html). AISTATS
</div>
