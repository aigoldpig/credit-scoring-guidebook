---
title: "실무 권장 — 상관 변수 해석 전략"
---

# 5. 실무 권장 — 상관 변수 해석 전략

> 이론적으로 완벽한 방법은 없다. 실무에서는 질문을 먼저 정하고, 방법을 명시하고, 두 방향의 결과를 비교하는 것이 현실적인 대응이다.

1~4절의 내용을 신용평가 모형 해석 실무의 절차로 정리한다.

---

## 5.1 먼저 질문을 정한다

상관이 있는 데이터에서는 하나의 숫자로 "모형에 충실"과 "데이터에 충실"을 동시에 만족할 수 없다([1절](observational-vs-interventional.md)). 따라서 해석을 시작하기 전에 **무엇을 묻는지** 정해야 한다.

| 질문 | 성격 | 적합한 방법 |
|---|---|---|
| 모형이 어떤 변수를 쓰고, 각 변수에 어떤 방향·크기로 반응하는가? (모형검증, 효과 방향 점검, 거절 사유) | 모형에 충실 | interventional SHAP, product measure fANOVA, ALE |
| 모형 입력에서 변수 하나만 바꾸면 점수가 얼마나 변하는가? (what-if) | 모형에 충실 | interventional SHAP, PDP·ICE (외삽 구간 주의) |
| 이 값을 가진 고객군은 실제로 어떤 점수를 받는가? (고객군·포트폴리오 설명) | 데이터에 충실 | observational 기대값(M-plot), 구간별 평균 예측 |
| 데이터가 있는 영역에서 효과를 주효과·교호작용으로 나누면? | 데이터 분포 반영 | joint measure fANOVA (반복 centering) + 교차공분산 보고 |

주의할 점이 하나 있다. "변수 하나만 바꾸면"이라는 what-if는 **모형의 반응**이지 현실의 인과효과가 아니다. 현실에서는 업권별 대출 건수를 하나 바꾸면 전체 대출 건수도 함께 바뀌듯이, 변수 하나만 독립적으로 움직이는 일은 드물다. interventional 결과를 고객에게 "이것만 바꾸면 점수가 오른다"는 식으로 그대로 안내해서는 안 된다.

## 5.2 양쪽 결과를 비교한다

상관이 강한 변수군이 있는 모형이라면, 하나의 방법만 보지 말고 **두 방향의 결과를 함께 계산해 비교**한다.

- SHAP: interventional(배경 데이터 지정) 값과, 상관을 반영하는 observational 값(조건부 기대값의 근사)을 함께 계산한다. path-dependent TreeSHAP은 상관을 트리 구조에 따라 일부만 반영하므로 observational 값을 대신하지 못한다는 점에 유의한다([3절](shap-two-modes.md)).
- fANOVA: product measure와 joint measure로 각각 분해한다. joint measure는 반복 centering으로 구하고, 조건부 평균을 재귀 공식에 그대로 넣는 방식은 쓰지 않는다([2절](joint-vs-product.md)).

비교 결과는 이렇게 읽는다.

- **두 결과가 비슷하면**: 그 변수의 해석은 상관 구조에 크게 흔들리지 않는다. 어느 쪽을 보고해도 결론이 같다.
- **두 결과가 크게 다르면**: 그 변수가 다른 변수와 효과를 나눠 갖고 있다는 신호다. [3절](shap-two-modes.md)의 예시에서 카드론의 SHAP 값은 가치함수에 따라 0.325~0.850으로 갈렸다. 이런 변수는 개별 값보다 **상관된 변수군 단위**로 해석하는 것이 안전하다.

과거 개발했던 XGBoost 기반 신용평가모형에서는 두 측도의 상위 변수 중요도 순위가 대체로 비슷했고, 차이는 상관이 강한 일부 변수에 집중되었다. 즉, 전체 그림은 유지되지만 **개별 변수의 몫**은 측도에 따라 달라질 수 있다.

## 5.3 Cross-covariance를 진단한다

fANOVA로 분산 분해를 보고할 때는 교차공분산 항을 함께 계산한다.

1. **전체 비중 확인**: 데이터 분포 위에서 \(\sum_{S \neq V} \text{Cov}(f_S, f_V)\)가 총 분산에서 차지하는 비중을 본다. 과거 개발했던 XGBoost 기반 신용평가모형에서는 이 비중이 전체 분산의 상당 부분을 차지했다([2절](joint-vs-product.md)). 이 항을 빼고 "주효과 몇 %, 교호작용 몇 %"만 보고하면 분산의 큰 몫이 설명에서 사라진다.
2. **쌍별 공분산 상위 목록**: \(\text{Cov}(f_i, f_j)\)가 큰 변수 쌍을 정렬해, 어떤 변수들이 효과를 나눠 갖고 있는지 확인한다. 대개 같은 신용 행태를 다른 각도에서 잰 변수들(업권별·기간별 대출 건수, 한도 대비 사용률과 잔액 등)이 상위에 온다.
3. **변수군 단위로 해석**: 공분산이 큰 변수들은 효과를 개별로 해석하지 말고 **합산 효과**(군에 속한 성분 또는 SHAP 값의 합)로 보거나, 대표 변수 하나를 남기고 나머지를 모형에서 빼는 것을 검토한다. 군 단위 합은 군 내부에서 몫을 어떻게 나누는지에 덜 민감하다.

## 5.4 EBM과 Purification

교호작용이 있는 모형을 해석 가능한 형태로 유지하려면, 학습 단계와 사후 단계를 함께 쓴다.

- **학습 단계**: EBM(Lou, Caruana & Gehrke, 2012; Lou et al., 2013)은 주효과를 먼저 학습한 뒤 교호작용 쌍을 잔차에 학습하는 2단계 구조로, 주효과가 교호작용으로 새어 나가는 것을 줄인다. 자세한 내용은 [EBM (GA²M)](../../part4_interpretation/ebm.md)을 참고한다.
- **사후 단계**: 학습이 끝난 모형에 purification을 적용해 fANOVA 정준형으로 옮긴다. 공개 구현으로 InterpretML의 purification 유틸리티(`interpret.utils.purify`)가 있다. 원리와 수렴 성질은 [부록 A 3절 Purification](../shap-fanova/purification.md)에서 다뤘다.
- **가중 선택을 명시**: purification 결과는 가중(측도)에 따라 달라진다. Lengerich et al. (2020, 6.1절)의 COMPAS 실험에서도, purification 여부와 가중 선택에 따라 main effect의 크기가 크게 달라지고 때로는 부호까지 달라졌다. 보고서에는 어떤 가중(uniform, empirical, Laplace)을 썼는지 반드시 적는다.

## 5.5 단조 제약을 활용한다

단조 제약(monotone constraint)은 상관 문제를 직접 해결하지는 않지만, 해석의 안정성을 높인다. 학습 시 어떤 변수에 단조 제약을 걸면, **다른 변수의 값을 어떻게 고정하든** 그 변수에 대해 모형 출력이 단조가 된다. 그 결과:

- **product measure fANOVA의 주효과, PDP, ALE**: 단조 함수들의 평균(또는 단조 증가분의 누적)이므로 **반드시 단조**다.
- **joint measure fANOVA의 주효과**: 가중이 변수 값마다 달라 '단조 함수들의 고정 가중 평균'이 아니므로 단조가 **보장되지 않는다.**
- **observational 기대값**: 상관된 변수의 효과가 섞여 방향이 반대로 나올 수도 있다.
- **SHAP dependence plot**(여러 고객의 SHAP 값을 변수 값 순으로 늘어놓은 그림): 고객마다 다른 변수 값이 달라 교호작용의 몫이 다르므로 단조 곡선으로 보이지 않을 수 있다. 다만 interventional SHAP은 한 고객의 다른 변수를 고정한 채 그 변수만 바꾸면 단조로 변한다(배경 데이터 위에서 단조 함수를 평균한 값이기 때문이다). observational·path-dependent SHAP은 이것도 보장되지 않는다.
- **교호작용 표 단독**: 주효과에 더하는 보정값이라 +와 −가 섞이는 것이 정상이다. 방향은 주효과와 더한 표로 읽는다.

즉 단조 제약 모형이라면, 효과 방향은 product measure 주효과·PDP·ALE나 고객 단위의 interventional SHAP처럼 단조가 보장되는 결과로 점검하는 것이 일관적이다. 모형을 점수표로 옮길 때 방향이 깨질 수 있는 경우(깊은 모형의 3-way 이상 항 절단, joint 가중 분해, 항별 반올림)는 [단조 제약은 어디까지 지켜 주나](../../part4_interpretation/interpretability.md#monotone-guarantee)에 정리했다. 이 가이드북이 해석 가능한 ML의 조건으로 제시하는 "교호작용 2-way 이하, 단조 제약, 점수표 환산 가능"([해석 가능성](../../part4_interpretation/interpretability.md))은 이런 점검을 가능하게 하는 최소 조건이기도 하다.

## 5.6 신용평가에서의 실무 권장

| 상황 | 권장 |
|---|---|
| **모형검증 — 모형이 어떤 변수를 쓰고 어떤 방향으로 반응하는가** | interventional SHAP 또는 product measure fANOVA를 기준으로 한다. observational 방식은 모형이 쓰지 않는 변수에도 기여를 배분하므로([4절](independence-extrapolation.md)), "모형이 이 변수를 쓴다"는 판단의 근거로 쓰지 않는다. |
| **거절 사유 등 개별 고객 설명** | interventional SHAP을 기본으로 하되, 배경 데이터(학습 데이터 표본 등)를 명시한다. path-dependent TreeSHAP을 쓴다면 트리 구조에 따라 상관된 변수 간 몫이 달라질 수 있음을 인지한다. 상관이 강한 변수군은 군 단위로 설명하는 것을 검토한다. |
| **고객군·포트폴리오 설명** | observational 기대값(구간별 실제 평균 예측)을 쓴다. 이것은 "모형의 반응"이 아니라 "데이터에서 관찰되는 점수 분포"임을 명시한다. |
| **효과 곡선 보고** | 상관이 강한 변수는 PDP 대신 ALE를 함께 제시하거나, PDP를 데이터 밀도가 있는 구간으로 한정한다. |
| **분산 분해 보고** | 측도(product/joint)를 명시하고, 교차공분산 항을 함께 보고한다. |
| **문서화** | 모형검증 문서에 사용한 가치함수·배경 데이터·측도·가중을 적고, 두 방식의 결과가 크게 다른 변수는 병기한다. |

어느 경우든 핵심은 같다. **"이 변수의 효과"는 측도를 정해야 비로소 정의된다.** 측도를 명시하지 않은 해석 결과는 재현할 수도, 검증할 수도 없다.

---

<div class="source-ref" markdown>
**참고**

- Lou, Caruana, Gehrke (2012). [Intelligible models for classification and regression](https://doi.org/10.1145/2339530.2339556). KDD
- Lou, Caruana, Gehrke, Hooker (2013). [Accurate intelligible models with pairwise interactions](https://doi.org/10.1145/2487575.2487579). KDD
- Lengerich, Tan, Chang, Hooker, Caruana (2020). [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html). AISTATS
- Chen, Janizek, Lundberg, Lee (2020). [True to the Model or True to the Data?](https://arxiv.org/abs/2006.16234) arXiv:2006.16234
- Apley, Zhu (2020). [Visualizing the effects of predictor variables in black box supervised learning models](https://doi.org/10.1111/rssb.12377). *JRSS-B* 82(4)
- [InterpretML purify API](https://interpret.ml/docs/python/api/purify.html)
</div>
