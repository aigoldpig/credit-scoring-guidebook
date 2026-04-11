---
title: "실무 권장 — 상관 변수 해석 전략"
---

# 5. 실무 권장 — 상관 변수 해석 전략

> 이론적 완벽함은 없다. 실무에서는 어떻게 대응해야 하는가?

!!! warning "작성 중"
    이 페이지는 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

## 다룰 내용

### 5.1 양쪽 결과를 비교하라

- Observational(A-1/B-1)과 Interventional(A-2/B-2) 모두 계산
- 두 결과가 유사하면: 변수 상관이 해석에 크게 영향을 주지 않음 → 안심
- 두 결과가 크게 다르면: 해당 변수군의 상관이 해석을 왜곡하고 있음 → 주의 필요
- 우리 실험에서 변수 순위 Top 10은 대부분 유사, 미세 차이만 존재

### 5.2 Cross-Covariance 진단

- 전체 분산 대비 Cross-cov 비율 확인 (우리 모형: 48~52%)
- 쌍별 Cov(f_i, f_j) Top 20 확인 → 어떤 변수 쌍이 해석을 흐리는지 식별
- 상관이 높은 변수 쌍: 합산 효과로 보거나, 대표 변수 1개로 해석

### 5.3 EBM과 Purification

- EBM (Lou et al., 2012, 2013): 학습 구조 자체가 main + 2-way 분리
- InterpretML `purify()`: 학습 후 사후 정제
- Lengerich et al. (2020): purification이 부호 반전을 방지한 COMPAS 사례
- 가이드북 부록 A-3 [Purification](../shap-fanova/purification.md)과 연결

### 5.4 단조성 제약 (Monotone Constraints)

- 상관 문제와 직접 관련은 아니지만, 해석 안정성에 기여
- 단조 제약 걸면 shape function이 단조 → 상관에 의한 비직관적 패턴 방지
- 가이드북 Part 4 해석 가능성 3대 조건과 연결: bin 5개 이하, 2-way, 단조성

### 5.5 신용평가에서의 실무 권장

- 규제 관점: "모형이 어떤 변수를 쓰는가" → Observational이 적합
- 인과 관점: "변수 하나를 바꾸면 스코어가 얼마나 변하는가" → Interventional이 적합
- 현업 설명 시: 둘의 차이를 인지하고, 해석 목적을 명시할 것
- **모형검증 보고서**에는 두 방식의 결과를 병기하는 것을 권장

---

<div class="source-ref" markdown>
**참고**

- Lou et al. (2013). *Accurate Intelligible Models with Pairwise Interactions.* KDD
- Lengerich et al. (2020). *Purifying Interaction Effects with the Functional ANOVA.* AISTATS
- Covert, Lundberg, Lee (2020). *Understanding Global Feature Contributions With Additive Importance Measures.* NeurIPS
</div>
