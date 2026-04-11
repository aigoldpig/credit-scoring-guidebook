---
title: "SHAP의 두 모드"
---

# 3. SHAP의 두 모드

> TreeSHAP에는 두 가지 feature_perturbation 모드가 있으며, fANOVA의 두 measure와 정확히 대응된다.

!!! warning "작성 중"
    이 페이지는 현재 작성 중이며, 내용이 추가·변경될 수 있습니다.

## 다룰 내용

### 3.1 tree_path_dependent (기본값)

- 변수가 coalition에 없을 때, 트리의 양쪽 자식으로 내려가되 **학습 시 샘플 비율**로 가중
- 조건부 분포를 트리 구조가 암묵적으로 인코딩
- background 데이터 불필요
- **fANOVA Joint measure에 대응**

### 3.2 interventional

- coalition에 없는 변수를 background 데이터에서 **독립적으로** 샘플링
- 변수 간 상관을 끊음 (Product measure)
- background 데이터 필요
- **fANOVA Product measure에 대응**

### 3.3 대응 관계 정리

| fANOVA | SHAP | Measure | 가정 |
|---|---|---|---|
| A-1 / B-1 | tree_path_dependent | Joint | 상관 반영 |
| A-2 / B-2 | interventional | Product | 독립 가정 |

### 3.4 Lundberg의 입장 — GitHub 인용

- [Issue #1098](https://github.com/slundberg/shap/issues/1098): `tree_path_dependent` vs `interventional` 두 모드의 차이 설명
- [Issue #1366](https://github.com/slundberg/shap/issues/1366): TreeExplainer data parameter — Hugh Chen(SHAP contributor)의 상세 설명
- [Issue #288](https://github.com/slundberg/shap/issues/288) (2018): "Accounting for feature dependence means you don't evaluate your model 'away from the manifold of the training data'."
- [Discussion #1538](https://github.com/shap/shap/discussions/1538): "you cannot be both true to the data and true to the model"
- **결론**: 목적에 따라 선택해야 함. 둘 다 구현한 것이 Lundberg의 답

### 3.5 상관 변수에서 SHAP 값이 이상해 보이는 이유

- [Issue #1120](https://github.com/slundberg/shap/issues/1120): "Correlation bias occurs because of how the ML algorithm trains the model, not because of how SHAP estimates feature importance."
- [Issue #1731](https://github.com/slundberg/shap/issues/1731): 상관된 두 변수의 SHAP 값이 반대 부호로 상쇄되는 현상
- 이것은 SHAP의 버그가 아니라, 모형이 상관 변수를 활용하는 방식의 반영

---

<div class="source-ref" markdown>
**참고**: Lundberg et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence 2:56-67
</div>
