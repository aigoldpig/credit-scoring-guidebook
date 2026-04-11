---
title: "두 가지 기대값"
---

# 1. 두 가지 기대값 — Observational vs Interventional

> 같은 질문 "변수 x가 a일 때 예측값은?"에 대해, 두 가지 서로 다른 답이 존재한다.

!!! warning "작성 중"
    이 페이지는 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

## 다룰 내용

### 1.1 Observational (관측적) 조건부 기대값

- \( E[f(X) \mid X_j = a] \) — "x_j가 a인 사람들의 평균 예측"
- 다른 변수의 **조건부 분포** \( P(X_{-j} \mid X_j = a) \) 를 사용
- 상관 구조가 반영됨: x_j가 낮은 사람은 x_k도 낮은 경향이 그대로 들어감
- **장점**: 실제 데이터 분포 위에서만 평가, 비현실적 조합 없음
- **단점**: x_j 자체의 순수 효과와 상관 변수의 효과가 뒤섞임

### 1.2 Interventional (개입적) 조건부 기대값

- \( E[f(X) \mid do(X_j = a)] \) — "x_j만 a로 고정하고 나머지는 전체 분포 그대로"
- 다른 변수의 **주변 분포** \( P(X_{-j}) \) 를 사용 (독립 가정)
- 상관 구조가 끊어짐: x_j와 x_k의 관계를 무시
- **장점**: 순수한 x_j 자체의 효과만 분리
- **단점**: 실제 존재하지 않는 변수 조합(소득 300만 + 부채 3억)도 평가

### 1.3 수치 예시

- depth-2 트리 1개를 가정한 간단한 수치 예시로 Observational vs Interventional 차이 시연

### 1.4 "True to the Model" vs "True to the Data"

- Chen, Lundberg, Lee (2020) 논문의 핵심 프레임
- Lundberg GitHub Discussion #1538 인용:
  > "In the presence of correlated features, you cannot be both true to the data and true to the model."

---

<div class="source-ref" markdown>
**참고**: Chen, Lundberg, Lee (2020). *True to the Model or True to the Data?* arXiv:2006.16234
</div>
