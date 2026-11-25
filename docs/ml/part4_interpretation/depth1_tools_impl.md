# 도구 · 구현 · 요약

## 6.1 관련 기법과 오픈소스 도구

1-Depth GBM과 EBM은 "해석 가능한 ML" 생태계의 일부다. 같은 철학(가법 구조 + 제한된 교호작용)을 공유하는 기법과 도구를 정리한다.

### 관련 모형 계보

| 기법 | 구조 | 연도 | 핵심 저자/기관 | 교호작용 |
|------|------|:----:|--------------|:--------:|
| **GAM** (splines) | 가산적 smooth 함수 | 1986 | Hastie, Tibshirani | 없음 |
| **GBM with stumps** | 가산적 boosted stump | 2001 | Friedman | 없음 |
| **GA\(^2\)M** | GAM + 선택적 pairwise | 2013 | Lou, Caruana | 2-way |
| **EBM** | Cyclic boosting 기반 GA\(^2\)M 구현 | 2019 | Nori, Caruana (Microsoft) | 선택적 2-way |
| **NAM** | Feature별 신경망 | 2021 | Agarwal, Hinton, Caruana 외 | 없음 |
| **GAMI-Net** | 신경망 GAM + purified interactions | 2021 | Yang, Zhang, Sudjianto | 2-way (purified) |
| **NODE-GAM** | Oblivious Decision Tree 기반 GAM | 2022 | Chang, Caruana, Goldenberg | 없음 (GA\(^2\)M 변형 있음) |
| **Add-XGBoost** | 변수별 XGBoost shape function의 가법 결합 + Lasso로 가중치 조정 | 2025 | Zou, Xia, Lan | 2-way |

### 오픈소스 도구

| 라이브러리 | 설명 | 특징 |
|-----------|------|------|
| **InterpretML / EBM** | Microsoft Research의 해석 가능 ML 프레임워크 | GA\(^2\)M의 대표 구현. Cyclic boosting, outer bagging |
| **PiML** | 해석 가능 ML 개발 및 진단 도구 | **XGB1** (Depth-1), **XGB2** (Depth-2 + purification)을 내장 모형으로 지원 |
| **xBooster** | 부스팅 모형 → 스코어카드 변환기 | 부스팅 모형의 leaf 값을 전통 스코어카드 포인트로 변환 |
| **skorecard** | ING Bank의 스코어카드 파이프라인 | 전통 LR 스코어카드 벤치마크용. 구간화 + 로지스틱 회귀 |

<div class="source-ref">
출처: Sudjianto, A. et al. (2023). "PiML Toolbox for Interpretable Machine Learning Model Development and Diagnostics." arXiv:2305.04214.
<a href="https://arxiv.org/abs/2305.04214" target="_blank">arXiv</a> ·
<a href="https://github.com/SelfExplainML/PiML-Toolbox" target="_blank">GitHub (PiML)</a> ·
<a href="https://github.com/interpretml/interpret" target="_blank">GitHub (InterpretML)</a> ·
<a href="https://github.com/xRiskLab/xBooster" target="_blank">GitHub (xBooster)</a>
</div>

---

## 6.2 실무 파라미터 가이드

1-Depth GBM을 XGBoost·LightGBM으로 만들 때의 출발점이다. 파라미터의 일반적인 의미는 [하이퍼파라미터 튜닝](../part3_tree_ensemble/hyperparameter_tuning.md)을 따른다.

| 파라미터 | 권장 값 | 이유 |
|---------|---------|------|
| `max_depth` | **1** | stump 구조 확보 (가법 구조 보장) |
| `num_leaves` (LightGBM) | **2** | depth=1이면 leaf는 반드시 2개 |
| 단조 제약 | **변수별 지정** | [Boosting 심화 — 단조 제약](../part3_tree_ensemble/boosting_advanced.md) 참조. WoE 입력이면 모두 −1 |
| `learning_rate` | 일반 권장 범위(0.01~0.1)의 낮은 쪽 | stump 하나의 보정량이 작아야 곡선이 매끄럽다. 기준은 [학습률 표](../part3_tree_ensemble/hyperparameter_tuning.md) |
| 트리 수 | 탐색 단계에서 Early Stopping으로 결정 | 최종 모형은 정한 트리 수를 완주 |
| `colsample_bytree` | 1.0에서 시작해 비교 | 상관된 변수 사이에서 효과가 한쪽으로 쏠린다면(concurvity), 열 샘플링으로 다른 변수에도 기회를 주는 것이 도움이 될 수 있다. 반대로 곡선이 재학습마다 크게 흔들리면 1.0이 낫다. 재학습 안정성을 보고 정한다 |
| `subsample` | **0.7 ~ 0.9** | row subsampling은 과적합 방지에 유효 |
| `reg_alpha` / `reg_lambda` | **0.1 ~ 1.0** | leaf weight 크기를 제어하여 개별 stump의 영향력을 제한 |
| 히스토그램 bin 수 | 필요 시 축소 | 분할점 후보를 줄여 곡선의 계단 수를 통제. 운영 평점표 관행(변수당 5개 안팎 구간)에 맞추려면 최종 점수표에서 인접 구간을 묶는다. 구간 수는 [해석 가능 조건](interpretability.md)이 아니다 |

!!! note "변수 수 관리"
    1-Depth GBM도 전통 스코어카드와 마찬가지로 **최종 변수를 10~20개 내외로 관리**하는 것이 바람직하다. Feature importance로 상위 변수를 선별한 뒤, 해당 변수만으로 재학습하는 2단계 접근이 실무적이다. 강하게 상관된 변수는 이 단계에서 하나만 남긴다.

---

## 6.3 실무 적용 시나리오

### 시나리오 1: 전통 스코어카드의 대안 모형

전통 LR 스코어카드의 대안으로 1-Depth GBM(또는 EBM)을 검토한다. 가법 구조이므로 변수별 효과를 점수표 형태로 제시할 수 있다.

- **적합한 경우**: 변수의 비선형 효과가 커서 수동 구간화로 놓치는 정보가 많고, 교호작용은 크지 않은 데이터
- **주의할 점**: 감독·검증 부서가 요구하는 문서화(구간 정의, 변수 선정 근거, 안정성 검증)를 전통 스코어카드 수준으로 갖춰야 한다. 성능 향상이 OOT에서 유지되는지 확인한다

### 시나리오 2: 챌린저 모형

운영 중인 전통 모형(챔피언) 대비 1-Depth GBM(챌린저)의 성능을 OOT에서 비교한다. 교체 근거를 확보하거나, 반대로 "현재 데이터에서는 비선형 이득이 크지 않다"는 것을 확인하는 데도 쓴다.

### 시나리오 3: 변수 탐색 도구

Full GBM을 바로 만들기 전에, 1-Depth GBM으로 **각 변수의 비선형 효과를 탐색**한다. WoE Classing의 구간 설정에도 참고할 수 있다 — "1-Depth GBM의 효과 곡선이 꺾이는 지점"이 자연스러운 Classing 경계 후보가 된다. 이 용도에서는 성능보다 곡선의 모양이 중요하므로 가장 부담 없이 쓸 수 있는 시나리오다.

### 시나리오 4: ML로 탐색, 전통으로 배포

Fahner (2018)의 transmutation 전략. Full GBM으로 패턴을 발견하고, 그 인사이트를 단조성 제약이 있는 스코어카드로 변환하여 배포한다([1-Depth GBM 스코어카드](depth1_gbm.md) 3.4절).

---

## 6.4 참고 자료

| 자료 | 유형 | 핵심 내용 |
|------|:----:|----------|
| Hastie & Tibshirani (1986). "Generalized Additive Models." *Statistical Science* | 논문 | GAM 프레임워크의 원조 |
| Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* | 논문 | GBM 원논문. 트리 크기와 교호작용 수준의 관계 설명 |
| Lou et al. (2012). "Intelligible Models for Classification and Regression." *KDD* | 논문 | GAM 적합법 비교 — 얕은 트리의 bagging + boosting이 가장 정확 |
| Lou et al. (2013). "Accurate Intelligible Models with Pairwise Interactions." *KDD* | 논문 | GA\(^2\)M — 2-way 교호작용 추가, FAST |
| Caruana et al. (2015). "Intelligible Models for HealthCare." *KDD* | 논문 | 해석 가능 ML의 필요성을 보여준 대표 사례 |
| Fahner (2018). "Developing Transparent Credit Risk Scorecards More Effectively." *DATA ANALYTICS 2018* | 논문 | FICO의 transmutation 전략 |
| Nori et al. (2019). "InterpretML." arXiv:1909.09223 | 논문/도구 | EBM 구현체와 InterpretML 프레임워크 |
| Lengerich et al. (2020). "Purifying Interaction Effects with the Functional ANOVA." *AISTATS* | 논문 | Main effect / interaction 분리 알고리즘 |
| Agarwal et al. (2021). "Neural Additive Models." *NeurIPS* | 논문 | 신경망 기반 GAM (NAM) |
| Yang, Zhang & Sudjianto (2021). "GAMI-Net." *Pattern Recognition* | 논문 | Purified interaction이 있는 신경망 GAM |
| Chang et al. (2021). "How Interpretable and Trustworthy are GAMs?" *KDD* | 논문 | 여러 GAM 적합법이 같은 데이터에서 서로 다른 shape function을 내는 문제를 비교 |
| Sudjianto et al. (2023). "PiML Toolbox." arXiv:2305.04214 | 논문/도구 | XGB1, XGB2를 해석 가능 모형으로 내장 |
| Zou, Xia & Lan (2025). "Interpretable credit scoring based on an additive extreme gradient boosting." *Chaos, Solitons & Fractals*, 194, 116216 | 논문 | 변수별 XGBoost shape function을 가산 결합하고 2차 교호작용과 Lasso 가중치 조정을 더한 신용평가 기법 |
| Moody's Analytics. "Automating Interpretable ML Scorecards." | 백서 | 해석 가능 모형 vs 비제약 모형 성능 비교 |

### 온라인 가이드

| 자료 | 설명 |
|------|------|
| [PiML --- EBM 가이드](https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/models/ebm.html) | PiML에서 EBM 학습, shape function 시각화, purification 적용 방법 |
| [InterpretML GitHub](https://github.com/interpretml/interpret) | EBM 구현체 소스 코드 및 예제 노트북 |

---

## 6.5 요약

| | 전통 LR | 1-Depth GBM | EBM (GA\(^2\)M) | Full GBM |
|---|:---:|:---:|:---:|:---:|
| **구조** | \(\beta_0 + \sum \beta_j \cdot \text{WoE}_j\) | \(F_0 + \sum f_j(x_j)\) | \(F_0 + \sum f_j + \sum f_{jk}\) | \(F_0 + \eta \sum h_t(\mathbf{x})\) |
| **해석** | 점수표 | 효과 곡선 | 효과 곡선 + 교호작용 | SHAP 등 사후 설명 |
| **비선형** | 구간화(수동) | 자동(stump 합) | 자동 + pairwise | 자동(깊은 트리) |
| **교호작용** | 없음 | 없음 | 선택적 2-way | 자동 |
| **해석 조건 충족** | 충족 | 단조 제약 시 충족 | 단조 처리 시 충족(운영 평점표로 쓰려면 bin 정리 필요) | 사후 요약만 가능 |

가법 모형 계열(1-Depth GBM, EBM)은 전통 스코어카드와 Full GBM 사이의 선택지다. 수동 구간화 없이 비선형 효과를 학습하면서도 변수별 효과를 곡선으로 보여줄 수 있다. 대신 교호작용을 포기하거나(1-Depth GBM) 제한해야 하고(EBM), 상관된 변수의 효과가 임의로 나뉘는 문제와 해석 조건(단조성, 점수표 환산)과 운영 평점표의 구간 수 관행을 맞추기 위한 추가 작업이 따른다. 어느 쪽이 나은지는 데이터의 교호작용 크기와 조직의 검증 요구 수준에 따라 다르므로, OOT 성능과 해석 검토 결과를 함께 놓고 정한다.

!!! tip "다음 페이지"
    [ML 해석을 고민한 기록](shap_in_practice.md) --- SHAP 실전의 벽, Counterfactual Explanation, 그리고 왜 SHAP은 교과서가 되고 fANOVA는 아닌가.
