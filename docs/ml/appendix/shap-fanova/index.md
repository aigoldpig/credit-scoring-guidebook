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
| 2 | [Functional ANOVA — 함수의 직교 분해](functional-anova.md) | fANOVA 정의, Zero Means · Orthogonality · Integrate-to-Zero 제약 |
| 3 | [Purification — 트리 모형의 사후 정제](purification.md) | Mass-Moving 알고리즘, 수렴 보장, California Housing · COMPAS 사례 |
| 4 | [SHAP vs fANOVA — 무엇이 다른가](shap-vs-fanova.md) | 분배 vs 분리, 교호작용 처리 차이, 분포 의존성, 실무 적용 |

---

<div class="source-ref" markdown>
**참고 문헌**

- Lengerich, Tan, Chang, Hooker, Caruana (2020). [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html) (AISTATS)
- Hooker (2004, 2007). Functional ANOVA를 ML 해석에 적용한 논문
- Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (NeurIPS)
</div>
