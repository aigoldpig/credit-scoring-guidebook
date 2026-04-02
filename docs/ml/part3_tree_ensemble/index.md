---
title: 트리 앙상블
---

# 트리 앙상블

> CART에서 XGBoost/LightGBM까지 -- 트리 기반 앙상블의 이론과 구조

단일 Decision Tree는 직관적이지만 불안정하다. 이 섹션에서는 단일 트리의 구조(CART)에서 출발하여, Bagging(Random Forest)과 Boosting(GBM, XGBoost, LightGBM)으로 이어지는 트리 기반 앙상블의 이론과 구조를 학습한다.

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [트리 기반 모델](tree_models.md) | CART의 분할 기준(Gini/Entropy)과 Pruning은 어떻게 동작하는가? |
| 2 | [앙상블 -- Bagging과 RF](ensemble.md) | Bootstrap, OOB, Feature Subsampling이 Variance를 어떻게 줄이는가? |
| 3 | [트리에서의 Bias-Variance](bias_variance_trees.md) | 하이퍼파라미터별로 Bias-Variance에 미치는 영향은? |
| 4 | [Boosting 기초](boosting_fundamentals.md) | AdaBoost에서 Gradient Boosting까지, 순차 학습의 핵심 아이디어는? |
| 5 | [Boosting 심화](boosting_advanced.md) | GBM의 B-V 관점, 트리 깊이와 교호작용의 관계는? |
| 6 | [XGBoost와 LightGBM](xgb_lgbm.md) | 정규화 목적함수, Histogram, GOSS, EFB 등 알고리즘 혁신은? |
| 7 | [하이퍼파라미터 튜닝](hyperparameter_tuning.md) | 핵심 파라미터, 단계적 튜닝 전략, Optuna 활용법은? |

!!! tip "다음 섹션"
    트리 앙상블의 이론과 구조를 마쳤다면, [해석과 설명](../part4_evaluation/index.md)에서 SHAP, Surrogate Model, 1-Depth GBM/EBM을 학습한다.
