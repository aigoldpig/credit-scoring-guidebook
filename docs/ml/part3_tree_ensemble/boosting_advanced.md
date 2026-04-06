# Boosting 심화

## 5.1 GBM의 Bias-Variance 관점

| 요소 | 역할 |
|------|------|
| **얕은 트리** (\(\text{depth}=3\sim6\)) | 개별 학습기의 Variance를 낮게 유지 |
| **순차 학습** | 라운드를 거듭하며 **Bias를 줄여나감** |
| **Learning Rate** (\(\eta\)) | 각 스텝의 보정량을 제한하여 **Variance 폭주 방지** |
| **Early Stopping** | Validation 성능이 악화되기 시작하면 학습 중단 |

> Bagging(RF)이 "강한 트리 + 평균화 → Variance ↓"였다면,
> Boosting(GBM)은 **"약한 트리 + 순차 보정 → Bias ↓"**이다.

둘의 접근 방향은 정반대지만, 목표는 같다 — Bias와 Variance 모두 낮은 **똑똑한 모형**을 만드는 것.

---

## 5.2 트리 깊이와 교호작용의 관계

GBM에서 개별 트리의 **깊이(depth)**는 모형이 포착할 수 있는 **교호작용의 차수**를 결정한다.

| Depth | 트리당 최대 변수 | 교호작용 차수 | 해석 가능성 |
|:-----:|:----------:|:-----------:|:----------:|
| **1** | 1개 | **없음** (GAM) | 매우 높음 |
| **2** | 2개 | 2-way | 높음 |
| **3** | 3개 | 3-way | 보통 |
| **5~6** | 5~6개 | 고차 | 낮음 (SHAP 필수) |

Depth = 1인 트리(stump)는 변수 하나만 사용하므로, stump의 합산은 **GAM(Generalized Additive Model)**과 동치가 된다 — 교호작용이 수학적으로 불가능하다. 반면 depth가 깊어질수록 고차 교호작용을 포착하지만 해석이 어려워진다. 이 트레이드오프와 그 실무적 의미는 [1-Depth GBM 스코어카드](../part4_interpretation/depth1_gbm.md)에서 상세히 다룬다.

---

## 5.3 요약

| 알고리즘 | 핵심 | 시대 |
|---------|------|------|
| **AdaBoost** | 틀린 샘플에 가중치 ↑ | 1997 |
| **Gradient Boosting** | 잔차(negative gradient)를 순차 학습 | 2001 |

$$
\boxed{F_T(x) = F_0(x) + \eta \sum_{t=1}^{T} h_t(x)}
$$

- 각 \(h_t\)는 이전 모형의 **실수를 보정**하는 얕은 트리
- 손실 함수만 바꾸면 회귀/분류/랭킹 등 어떤 문제에도 적용 가능
- Learning Rate \(\eta\)와 트리 수 \(T\)의 균형이 핵심

!!! tip "다음 섹션"
    Gradient Boosting의 원리를 이해했으니, 이를 **고속·고성능으로 구현**한 [XGBoost와 LightGBM](xgb_lgbm.md)의 구체적인 최적화 전략을 살펴본다.
