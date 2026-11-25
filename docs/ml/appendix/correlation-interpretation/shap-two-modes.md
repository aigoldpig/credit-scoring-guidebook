---
title: "SHAP의 두 모드"
---

# 3. SHAP의 두 모드

> SHAP 값은 "가치함수"를 무엇으로 정하느냐에 따라 달라진다. TreeSHAP의 두 모드는 fANOVA의 두 측도와 비슷한 선택이지만, 정확히 같지는 않다.

SHAP은 변수 집합 \(S\)의 "가치" \(v(S)\)를 정한 뒤, 각 변수가 coalition에 들어갈 때 가치가 얼마나 늘어나는지를 모든 순서에 대해 평균한다. 두 변수라면

$$
\phi_1 = \tfrac{1}{2}\big[(v(\{1\}) - v(\emptyset)) + (v(\{1,2\}) - v(\{2\}))\big]
\tag{1}
$$

이고, \(\phi_2\)도 같은 방식이다. Shapley 공리들은 \(v\)가 주어진 뒤의 **분배 규칙**만 정할 뿐, \(v(S)\) — "\(S\)만 알 때의 예측" — 를 어떻게 정의할지는 정하지 않는다. [1절](observational-vs-interventional.md)의 두 가지 기대값이 그대로 두 가지 가치함수가 되고, 트리 모형에서는 여기에 TreeSHAP 고유의 path-dependent 가치함수가 더해진다.

---

## 3.1 Interventional SHAP

$$
v_{\text{int}}(S) = \mathbb{E}_{\mathbf{X}_{-S} \sim \mathcal{B}}\big[f(\mathbf{x}_S, \mathbf{X}_{-S})\big]
\tag{2}
$$

coalition 밖의 변수를 **배경 데이터 \(\mathcal{B}\)**의 값으로 채워 평균한다. TreeSHAP의 interventional 모드는 배경 데이터의 각 행에 대해 이 계산을 트리 구조로 정확히 수행한다. 배경 데이터의 한 행에서 \(\mathbf{x}_{-S}\)를 **통째로** 가져오므로, \(S\)와 \(-S\) 사이의 상관만 끊기고 \(-S\) 내부의 상관은 유지된다.

**fANOVA와의 정확한 관계.** 배경 분포가 변수 독립(product measure)이면, interventional SHAP은 product measure fANOVA 성분을 교호작용 차수로 나눠 배분한 것과 정확히 같다 (Herren, Hahn 외, arXiv:2208.09970; Hiabu, Meyer & Wright, 2023):

$$
\phi_j(\mathbf{x}) = \sum_{S \ni j} \frac{f_S(\mathbf{x}_S)}{|S|}
\tag{3}
$$

실무에서는 배경 데이터로 학습 데이터(상관 있음)를 쓰는 경우가 많은데, 이때는 \(v(\emptyset)\)부터 실제 데이터의 평균 예측이 되어 식 (3)이 정확히 성립하지 않는다. 아래 예시에서 두 경우를 모두 계산한다.

## 3.2 Observational (조건부) SHAP

$$
v_{\text{obs}}(S) = \mathbb{E}\big[f(\mathbf{X}) \mid \mathbf{X}_S = \mathbf{x}_S\big]
\tag{4}
$$

[1절](observational-vs-interventional.md)의 식 (1)을 그대로 가치함수로 쓴다. 상관을 반영하므로 데이터에 없는 조합을 평가하지 않지만, 정확히 계산하려면 모든 \(S\)에 대한 조건부 분포가 필요해 일반적으로 계산이 어렵다. 또 [4절](independence-extrapolation.md)에서 보듯이 모형이 쓰지 않는 변수에도 0이 아닌 값을 줄 수 있다. 식 (4)에 대응하는 fANOVA 성분은 없다 — [2절](joint-vs-product.md)의 joint measure fANOVA(Hooker 2007)와도 다른 양이다.

## 3.3 Path-dependent TreeSHAP (트리 경로 기반)

TreeSHAP의 또 다른 모드는 배경 데이터 없이 트리 자체의 정보만 쓴다. coalition 밖의 변수로 분할하는 노드를 만나면 **양쪽 가지를 모두 따라가면서**, 학습 때 각 가지로 흘러간 샘플 비율(cover)로 가중평균한다. TreeSHAP은 이 가치함수에 대한 Shapley 값을 다항 시간에 **정확히** 계산한다.

이 가치함수는 조건부 기대값 \(v_{\text{obs}}\)를 트리 구조로 근사한 것으로 볼 수 있다. 하지만 cover 비율은 **트리 경로의 위쪽 분할 조건에만** 조건부이기 때문에, 어떤 변수를 먼저 분할했느냐에 따라 어떤 상관은 반영되고 어떤 상관은 무시된다. 그 결과 path-dependent SHAP은 일반적으로 observational도 interventional도 아니며, **같은 예측 함수라도 트리 구조가 다르면 값이 달라진다.** 한편 모형이 분할에 쓰지 않은 변수는 어떤 경로에도 등장하지 않으므로, 그 변수의 SHAP 값은 항상 0이다.

`shap` 라이브러리의 TreeExplainer는 `feature_perturbation` 인자로 모드를 정한다. 현재 기본값인 "auto"에서는 배경 데이터를 넘기면 interventional, 넘기지 않으면 path-dependent(`tree_path_dependent`) 방식을 쓴다([공식 문서](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)). 따라서 같은 코드라도 **배경 데이터를 넘겼는지에 따라 다른 질문에 답하게 된다.** SHAP 기여자 Hugh Chen도 공개 이슈에서, TreeExplainer의 `data` 인자는 배경 데이터이며 이를 주지 않으면 path-dependent 버전이 쓰인다고 설명했다([shap issue #1366](https://github.com/shap/shap/issues/1366)).

---

## 3.4 수치 예시 — 같은 고객, 다섯 가지 SHAP 값

[1절](observational-vs-interventional.md)의 공통 예시(현금서비스 \(X_1\), 카드론 \(X_2\), 상관 0.6, 모형 \(f = 1.0\,x_1 + 0.5\,x_2 + 1.0\,x_1 x_2\))에서, 둘 다 이용한 고객 \(\mathbf{x} = (1, 1)\)을 설명한다. 이 고객의 예측은 2.5다.

모형 \(f\)는 depth-2 트리 두 가지로 똑같이 표현할 수 있다: 루트에서 \(X_1\)으로 먼저 나누는 트리, 그리고 \(X_2\)로 먼저 나누는 트리. 두 트리는 네 칸 모두에서 같은 값을 낸다.

| 가치함수 | \(v(\emptyset)\) | \(v(\{1\})\) | \(v(\{2\})\) | \(\phi_1\) (현금서비스) | \(\phi_2\) (카드론) | 합 |
|---|---:|---:|---:|---:|---:|---:|
| Observational (식 4) | 1.15 | 2.20 | 2.10 | 0.725 | 0.625 | 1.35 |
| Interventional, 배경 = 데이터 | 1.15 | 1.75 | 1.50 | 0.800 | 0.550 | 1.35 |
| Interventional, 배경 = 독립(product) | 1.00 | 1.75 | 1.50 | 0.875 | 0.625 | 1.50 |
| Path-dependent, \(X_1\) 먼저 분할한 트리 | 1.15 | 2.20 | 1.50 | 1.025 | 0.325 | 1.35 |
| Path-dependent, \(X_2\) 먼저 분할한 트리 | 1.15 | 1.75 | 2.10 | 0.500 | 0.850 | 1.35 |

(\(v(\{1,2\}) = 2.5\)는 모든 경우에 같다. 합은 \(v(\{1,2\}) - v(\emptyset)\)이다.)

읽는 법:

- **Interventional(배경 = 독립)과 fANOVA**: product measure fANOVA로 분해하면 \(f_0 = 1.0\), \(f_1(1) = 0.75\), \(f_2(1) = 0.5\), \(f_{12}(1,1) = 0.25\)다([2절](joint-vs-product.md)과 같은 방식으로 계산). 식 (3)에 넣으면 \(\phi_1 = 0.75 + 0.25/2 = 0.875\), \(\phi_2 = 0.5 + 0.25/2 = 0.625\)로, 표의 값과 정확히 일치한다.
- **Interventional(배경 = 데이터)**: \(v(\emptyset)\)가 실제 데이터의 평균 1.15로 바뀌면서 SHAP 값이 0.800, 0.550이 된다. 실무에서 가장 흔한 설정이지만, 식 (3)의 "균등 분배"는 근사적인 직관으로만 맞는다.
- **Observational**: 카드론의 몫이 0.550에서 0.625로 커졌다. 카드론 이용자 대부분이 현금서비스도 이용하기 때문에, 현금서비스의 위험 일부가 카드론의 몫으로 넘어온다.
- **Path-dependent**: 같은 함수인데 트리 구조에 따라 카드론의 SHAP 값이 0.325에서 0.850까지 2.6배 차이가 난다. \(X_1\)을 먼저 분할한 트리에서는 \(v(\{1\})\)이 observational(2.20)과 같고 \(v(\{2\})\)는 interventional(1.50)과 같다. 루트의 \(X_1\) 분할은 상관을 반영하지 못하고, 두 번째 단계의 \(X_2\) 분할만 상관을 반영하기 때문이다.

### 모형이 쓰지 않는 변수

현금서비스만 쓰는 모형 \(g = 2.0\,x_1\)(트리가 \(X_1\)으로만 분할)으로 같은 고객을 설명하면:

| 가치함수 | \(\phi_1\) | \(\phi_2\) (카드론, 모형 미사용) |
|---|---:|---:|
| Observational | 0.70 | **0.30** |
| Interventional (배경 = 데이터) | 1.00 | 0.00 |
| Path-dependent | 1.00 | 0.00 |

observational SHAP은 모형이 보지도 않는 카드론에 0.30을 배분한다. interventional과 path-dependent는 0을 준다. 이것이 [4절](independence-extrapolation.md)의 Dummy 공리 문제다.

## 3.5 대응 관계 정리

| 가치함수 | 상관 처리 | 대응하는 fANOVA | 모형 미사용 변수 | 비고 |
|---|---|---|---|---|
| Interventional, 독립 배경 | 모두 끊음 | product measure — **정확히** 식 (3) | 0 | 외삽 발생 |
| Interventional, 데이터 배경 | \(S\)와 \(-S\) 사이만 끊음 | product measure와 근사적으로 대응 | 0 | 실무에서 가장 흔함, 외삽 발생 |
| Observational | 모두 반영 | 대응하는 성분 없음 | 0이 아닐 수 있음 | 정확한 계산이 어려움 |
| Path-dependent (TreeSHAP 기본) | 트리 경로 위쪽 조건만 반영 | 대응하는 성분 없음 | 0 | 트리 구조에 따라 값이 달라짐 |

흔히 "path-dependent = joint measure, interventional = product measure"라고 짝지어 설명하지만, 위 표에서 보듯이 정확한 대응은 **독립 배경의 interventional SHAP ↔ product measure fANOVA** 하나뿐이다. path-dependent는 상관을 일부만 반영하는 별개의 가치함수로 이해하는 것이 안전하다.

## 3.6 상관 변수에서 SHAP 값이 이상해 보이는 이유

상관이 강한 두 변수의 SHAP 값이 직관과 다르게 나오는 데에는 두 가지 원인이 겹친다.

1. **모형 자체가 효과를 임의로 나눠 가진다.** 두 변수가 거의 같은 정보를 담고 있으면, 트리 모형은 어느 쪽으로 분할해도 비슷한 성능을 얻는다. 학습 과정의 우연(분할 순서, 샘플링)에 따라 한 변수가 효과를 거의 다 가져가거나, 단조 제약이 없으면 두 변수가 반대 부호의 효과로 서로 상쇄하는 모형이 만들어지기도 한다(두 변수에 같은 방향의 단조 제약을 걸면 이런 역방향 학습은 막힌다). 이것은 SHAP의 계산 문제가 아니라 **모형이 그렇게 학습된 것**이며, interventional SHAP은 그 모형을 충실하게 보여줄 뿐이다.
2. **가치함수가 상관을 다루는 방식이 다르다.** 위 예시처럼 observational은 상관된 변수에 몫을 나눠주고, path-dependent는 트리 구조에 따라 일부만 나눠준다.

Lundberg는 상관을 반영하는 방식을 택하는 이유를 "학습 데이터의 분포(manifold)에서 벗어난 곳에서 모형을 평가하지 않기 위해서"라는 취지로 설명한 바 있다([shap issue #288](https://github.com/shap/shap/issues/288), 2018). 그리고 이후 [interpretable-ml-book issue #142](https://github.com/christophM/interpretable-ml-book/issues/142#issuecomment-565145918) 코멘트에서는 변수가 상관되어 있으면 모형과 데이터 모두에 충실할 수는 없다고 정리했다. shap 라이브러리가 두 모드를 모두 제공하는 것은 **목적에 따라 사용자가 고르라는 것**이며, 선택 기준은 [5절](practical-guide.md)에서 다룬다.

---

<div class="source-ref" markdown>
**참고**

- Lundberg et al. (2020). [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence* 2:56–67 — TreeSHAP
- Chen, Janizek, Lundberg, Lee (2020). [True to the Model or True to the Data?](https://arxiv.org/abs/2006.16234) arXiv:2006.16234
- Herren, Hahn 외. [Statistical Aspects of SHAP: Functional ANOVA for Model Interpretation](https://arxiv.org/abs/2208.09970). arXiv:2208.09970
- Hiabu, Meyer, Wright (2023). [Unifying local and global model explanations by functional decomposition of low dimensional structures](https://proceedings.mlr.press/v206/hiabu23a.html). AISTATS, PMLR 206
- [shap TreeExplainer 공식 문서](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
</div>
