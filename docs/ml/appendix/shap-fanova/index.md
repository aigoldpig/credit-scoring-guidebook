---
title: "SHAP과 Functional ANOVA"
---

# 부록 A: SHAP과 Functional ANOVA

> ML 모형의 해석에서 핵심이 되는 두 가지 수학적 프레임워크 — Functional ANOVA 분해와 SHAP의 이론적 기초, 차이, 그리고 Purification

!!! note "이 부록의 위치"
    [해석 가능성](../../part4_interpretation/interpretability.md), [SHAP 이론](../../part4_interpretation/shap_theory.md), [fANOVA 개념과 Purification](../../part4_interpretation/fanova_concepts.md)에서 다룬 실무적 내용의 **수학적 기초**에 해당한다. "왜 분해가 필요한가"부터 Purification 알고리즘까지를 정리한다.

## 목차

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [왜 분해가 필요한가 — Non-identifiability 문제](non-identifiability.md) | 제약 없는 분해의 비유일성, Boolean 예시, 실무 위험 |
| 2 | [Functional ANOVA — 함수의 직교 분해](functional-anova.md) | fANOVA 정의, 유일성을 보장하는 Integrate-to-Zero 조건, 그 결과인 Zero Means · Orthogonality · Variance Decomposition |
| 3 | [Purification — 트리 모형의 사후 정제](purification.md) | 트리 앙상블의 텐서 변환, Mass-Moving 알고리즘, 수렴 속도, 가중(측도) 선택, California Housing · COMPAS 사례 |
| 4 | [SHAP vs fANOVA — 무엇이 다른가](shap-vs-fanova.md) | 분배 vs 분리, SHAP과 fANOVA의 정확한 관계식, SHAP interaction value, 분포 의존성, 실무 적용 |

!!! warning "측도 가정"
    이 부록의 수식과 예시는 기본적으로 **변수 독립(product measure)**을 전제로 한다. 신용평가 변수처럼 상관이 강할 때 무엇이 달라지는지는 [부록 B: 변수 상관과 모형 해석](../correlation-interpretation/index.md)에서 다룬다.

---

<div class="source-ref" markdown>
**참고 문헌**

- Lengerich, Tan, Chang, Hooker, Caruana (2020). [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html) (AISTATS)
- Hooker (2004). [Discovering additive structure in black box functions](https://doi.org/10.1145/1014052.1014122) (KDD)
- Hooker (2007). [Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables](https://doi.org/10.1198/106186007X237892) (*JCGS* 16(3))
- Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (NeurIPS)
- Lundberg et al. (2020). [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9) (*Nature Machine Intelligence* 2)
</div>
