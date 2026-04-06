# 도구 · 구현 · 요약

## 6.1 관련 기법과 오픈소스 도구

1-Depth GBM은 "해석 가능한 ML" 생태계의 일부다. 같은 철학을 공유하는 기법과 도구를 정리한다.

### 관련 모형 계보

| 기법 | 구조 | 연도 | 핵심 저자/기관 | 교호작용 |
|------|------|:----:|--------------|:--------:|
| **GAM** (splines) | 가산적 smooth 함수 | 1986 | Hastie, Tibshirani | 없음 |
| **GBM with stumps** | 가산적 boosted stump | 2001 | Friedman | 없음 |
| **GA\(^2\)M** | GAM + 선택적 pairwise | 2013 | Lou, Caruana | 2-way |
| **EBM** | Round-robin cyclic boosting GAM | 2019 | Nori, Caruana (Microsoft) | 선택적 2-way |
| **NAM** | Feature별 신경망 | 2021 | Agarwal, Hinton, Caruana | 없음 |
| **GAMI-Net** | 신경망 GAM + purified interactions | 2021 | Yang, Sudjianto | 2-way (purified) |
| **NODE-GAM** | Oblivious Decision Tree GAM | 2022 | Chang, Caruana | 없음 |
| **Add-XGBoost** | Feature별 XGBoost + Lasso | 2025 | (신용평가 특화) | 2-way |

### 오픈소스 도구

| 라이브러리 | 설명 | 특징 |
|-----------|------|------|
| **InterpretML / EBM** | Microsoft Research의 해석 가능 ML 프레임워크 | GAM의 "정석" 구현. Round-robin 학습, inner/outer bagging |
| **PiML** | 해석 가능 ML 개발 및 진단 도구 | **XGB1** (Depth-1), **XGB2** (Depth-2 + purification)을 내장 모형으로 지원 |
| **xBooster** | GBM → 스코어카드 변환기 | XGBoost/LightGBM/CatBoost의 leaf weight를 전통 스코어카드 포인트로 변환 |
| **skorecard** | ING Bank의 스코어카드 파이프라인 | 전통 LR 스코어카드 벤치마크용. WoE 구간화 + 로지스틱 회귀 |

<div class="source-ref">
출처: Sudjianto, A. et al. (2023). "PiML Toolbox for Interpretable Machine Learning Model Development and Diagnostics." arXiv:2305.04214.
<a href="https://arxiv.org/abs/2305.04214" target="_blank">arXiv</a> ·
<a href="https://github.com/SelfExplainML/PiML-Toolbox" target="_blank">GitHub (PiML)</a> ·
<a href="https://github.com/interpretml/interpret" target="_blank">GitHub (InterpretML)</a> ·
<a href="https://github.com/xRiskLab/xBooster" target="_blank">GitHub (xBooster)</a>
</div>

---

## 6.2 실무 파라미터 가이드

### 핵심 파라미터

| 파라미터 | 권장 값 | 이유 |
|---------|---------|------|
| `max_depth` | **1** | stump 구조 확보 (GAM 보장) |
| `num_leaves` | **2** | depth=1이면 leaf는 반드시 2개 |
| `monotone_constraints` | **변수별 지정** | [개요 — 단조성](depth1_gbm.md) 참조 |
| `learning_rate` | **0.01 ~ 0.05** | 낮을수록 shape function이 부드럽고 안정적 |
| `n_estimators` | **1,000 ~ 5,000** | 낮은 learning_rate를 보상. early stopping과 병용 |
| `colsample_bytree` | **1.0 권장** | depth-1 stump은 변수 1개만 사용하므로, subsampling하면 shape function이 불안정해질 수 있다 |
| `subsample` | **0.7 ~ 0.9** | row subsampling은 과적합 방지에 유효 |
| `reg_alpha` / `reg_lambda` | **0.1 ~ 1.0** | leaf weight 크기를 제어하여 개별 stump의 영향력을 제한 |

!!! note "변수 수 관리"
    1-Depth GBM도 전통 스코어카드와 마찬가지로 **최종 변수를 10~20개 내외로 관리**하는 것이 바람직하다. Feature importance로 상위 변수를 선별한 뒤, 해당 변수만으로 재학습하는 2단계 접근이 실무적이다.

---

## 6.3 실무 적용 시나리오

### 시나리오 1: 규제 모형 대체

전통 LR 스코어카드의 대체 모형으로 1-Depth GBM을 제안. GAM 구조이므로 변수별 효과를 점수표처럼 제출할 수 있다.

### 시나리오 2: 챌린저 모형

운영 중인 전통 모형(챔피언) 대비 1-Depth GBM(챌린저)의 성능 우위를 OOT에서 검증하여 교체 근거를 확보.

### 시나리오 3: 변수 탐색 도구

Full GBM을 바로 만들기 전에, 1-Depth GBM으로 **각 변수의 비선형 효과를 탐색**한다. WoE Classing의 구간 설정에도 참고할 수 있다 — "1-Depth GBM의 효과 곡선이 꺾이는 지점"이 자연스러운 Classing 경계가 된다.

### 시나리오 4: FICO 방식 — ML로 탐색, 전통으로 배포

Fahner (2018)의 transmutation 전략. Full GBM으로 패턴을 발견하고, 그 인사이트를 단조성 제약이 있는 스코어카드로 변환하여 배포한다.

---

## 6.4 참고 자료

| 자료 | 유형 | 핵심 내용 |
|------|:----:|----------|
| Hastie & Tibshirani (1986). "Generalized Additive Models." *Statistical Science* | 논문 | GAM 프레임워크의 원조 |
| Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* | 논문 | GBM 원논문. Stump = additive model 명시 |
| Lou et al. (2012). "Intelligible Models for Classification and Regression." *KDD* | 논문 | Tree-based GAM이 최적 적합법임을 실증 |
| Lou et al. (2013). "Accurate Intelligible Models with Pairwise Interactions." *KDD* | 논문 | GA\(^2\)M — 2-way 교호작용 추가 |
| Caruana et al. (2015). "Intelligible Models for HealthCare." *KDD* | 논문 | 해석 가능 ML의 필요성을 보여준 대표 사례 |
| Fahner (2018). "Developing Transparent Credit Risk Scorecards." *Data Analytics* | 논문 | FICO의 transmutation 전략. **Best Paper Award** |
| Nori et al. (2019). "InterpretML." arXiv:1909.09223 | 논문/도구 | EBM 구현체와 InterpretML 프레임워크 |
| Lengerich et al. (2020). "Purifying Interaction Effects." *AISTATS* | 논문 | Main effect / interaction 분리 알고리즘 |
| Agarwal et al. (2021). "Neural Additive Models." *NeurIPS* | 논문 | 신경망 기반 GAM (NAM) |
| Yang et al. (2021). "GAMI-Net." *Pattern Recognition* | 논문 | Purified interaction이 있는 신경망 GAM |
| Chang et al. (2021). "How Interpretable and Trustworthy are GAMs?" *KDD* | 논문 | Tree-based GAM이 가장 신뢰할 수 있는 GAM |
| Sudjianto et al. (2023). "PiML Toolbox." arXiv:2305.04214 | 논문/도구 | XGB1, XGB2를 해석 가능 모형으로 내장 |
| (2025). "Interpretable credit scoring based on Add-XGBoost." *Chaos, Solitons & Fractals* | 논문 | Feature별 XGBoost를 가산적으로 결합하는 신용평가 특화 기법 |
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
| **구조** | \(\beta_0 + \sum \beta_j \cdot \text{WoE}_j\) | \(F_0 + \sum f_j(x_j)\) | \(F_0 + \sum f_j + \sum f_{jk}\) | \(F_0 + \sum h_t(\mathbf{x})\) |
| **해석** | 점수표 | 효과 곡선 | 효과 곡선 + 교호작용 | SHAP |
| **비선형** | 구간화(수동) | 자동(stump 합) | 자동 + pairwise | 자동(깊은 트리) |
| **교호작용** | 없음 | 없음 | 선택적 2-way | 자동 |
| **규제 수용성** | 매우 높음 | 높음 | 높음 | 제한적 |

1-Depth GBM은 전통과 ML 사이의 **징검다리**다. 전통의 해석 가능성과 ML의 자동 비선형 학습을 동시에 얻을 수 있다.

한 걸음 더 나아가면, EBM은 round-robin 학습과 선택적 교호작용으로 1-Depth GBM의 발전형이며, Full GBM과의 성능 격차를 상당 부분 좁힌다. PiML, xBooster 같은 도구 생태계도 성숙해지고 있다.

모든 상황에서 최선은 아니지만, **"왜 ML을 쓰는가?"에 대한 가장 설명하기 쉬운 답**이다.

!!! tip "다음 페이지"
    [ML 해석을 고민한 기록](shap_in_practice.md) --- SHAP 실전의 벽, Counterfactual Explanation, 그리고 왜 SHAP은 교과서가 되고 fANOVA는 아닌가.
