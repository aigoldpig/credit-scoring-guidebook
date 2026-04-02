# 도구 · 구현 · 요약

## 5.1 관련 기법과 오픈소스 도구

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

## 5.2 업계 적용 사례

### FICO의 접근: Transmutation

FICO의 Gerald Fahner가 제안한 **transmutation** 방법은:

1. **Tree Ensemble Model (TEM)으로 패턴 발견** — Full GBM으로 변수 간 비선형 관계를 탐색
2. **TEM의 인사이트를 제약 스코어카드로 변환** — 발견된 패턴을 단조성 제약이 있는 세분화된 스코어카드로 변환 (비선형 프로그래밍 사용)
3. **스코어카드를 배포** — 블랙박스 TEM이 아닌, 해석 가능한 스코어카드를 실전에 투입

이 접근은 "ML은 도구로만 사용하고, 최종 산출물은 전통 스코어카드"라는 실용적 전략이다. 이 논문은 Data Analytics 2018에서 **Best Paper Award**를 수상했다.

<div class="source-ref">
출처: Fahner, G. (2018). "Developing Transparent Credit Risk Scorecards More Effectively: An Explainable Artificial Intelligence Approach." <em>Data Analytics 2018</em>.
<a href="https://www.philadelphiafed.org/-/media/frbp/assets/events/2018/consumer-finance/fintech-2018/day-1/session_3_paper_3_fico_paper_gerald_fahner.pdf" target="_blank">PDF</a>
</div>

### KCB의 접근: 1-Depth GBM 특허 (2018)

한국에서도 이 아이디어가 독자적으로 발전했다. **코리아크레딧뷰로(KCB)**와 **서울대학교 산학협력단(김용대 교수 연구실)**이 공동 출원한 등록특허 **10-1851367** (2018)은, 1-Depth GBM을 스코어카드로 자동 변환하는 전체 파이프라인을 체계화한 것이다.

#### 특허가 지적하는 기존 방식(FICO/LR)의 한계

1. **시간 비용** — 후보 변수 300\~1,000개를 전문가가 일일이 분석·구간화해야 하므로, 개발에 많은 시간과 리소스가 소요
2. **단변량 구간화의 한계** — 변수를 독립적으로 구간화하면 다변량 모형에서는 최적 구간이 아닐 수 있음
3. **다중공선성에 의한 변수 제한** — 최종 변수가 10\~15개로 제한되어 정보 손실 발생
4. **전문가 개입에 의한 주관성** — 개발자에 따라 평가 결과가 달라질 수 있음

#### 4단계 자동 파이프라인

```
Training Data ──→ [S10] 1차 모형 ──→ [S20] 최적 모형 ──→ [S30] 신용평가모형 ──→ [S40] 스코어카드
                  (t개 스텀프)      (k개 선택)          (변수별 그룹핑)        (PDO·BASE 변환)
```

**S10 — 1차 모형 모델링**: depth=1 의사결정나무(스텀프) t개를 순차 학습. 변수별 모노톤 제약을 지정하여 도메인 지식을 반영.

**S20 — 최적 모형 선택**: 누적 변별력 지표(AUROC, K-S, AR, IV)로 오버피팅 직전인 k개 나무까지만 채택.

**S30 — 신용평가모형 변환**: 동일 변수의 스텀프들을 그룹핑, 각 split point를 구간 경계로 변환. 본질적으로 **GAM의 shape function을 구간별 상수(계단 함수)로 이산화**하는 것이며, 전통 WoE 스코어카드와 동일한 형태의 점수표가 된다.

**S40 — 스코어카드 생성**: PDO(Points to Double Odds)와 BASE를 반영하여 최종 점수 스케일로 변환.

!!! tip "이 특허가 의미하는 것"
    핵심은 새로운 알고리즘의 발명이 아니라, **1-Depth GBM → 해석 가능한 스코어카드**로의 자동 변환 파이프라인을 산업적으로 체계화한 데 있다. **한국의 주요 CB가 이 방법론을 특허로 보호할 만큼 실전적 가치를 인정했다**는 점이 중요하다.

<div class="source-ref">
출처: 강신형, 김용대. "신용도를 평가하는 방법, 장치 및 컴퓨터 판독 가능한 기록 매체." 등록특허 10-1851367 (2018). 특허권자: 코리아크레딧뷰로(주), 서울대학교 산학협력단.
</div>

---

## 5.3 실무 파라미터 가이드

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

## 5.4 구현 예시

### 1-Depth GBM 학습

```python
import lightgbm as lgb

params = {
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 1,          # 핵심: depth = 1
    'num_leaves': 2,          # depth=1이면 leaf는 2개
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 1.0,  # depth=1에서는 1.0 권장
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'monotone_constraints': [1, -1, 0, ...],  # 변수별 단조성 제약 (필수)
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### EBM (InterpretML) 학습

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(
    max_bins=256,
    interactions=10,         # 상위 10개 2-way 교호작용만 추가
    outer_bags=25,
    inner_bags=25,
    learning_rate=0.01,
    min_samples_leaf=2,
    max_rounds=5000
)
ebm.fit(X_train, y_train)

# 변수별 효과 곡선 시각화
from interpret import show
ebm_global = ebm.explain_global()
show(ebm_global)
```

### 변수별 효과 곡선 추출 (SHAP)

```python
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

# 1-Depth에서는 SHAP dependence = 변수의 partial effect
for feature in top_features:
    shap.dependence_plot(feature, shap_values, X_valid,
                         interaction_index=None)  # 교호작용 없으므로
```

### xBooster: GBM → 스코어카드 포인트 변환

```python
from xbooster import XBooster

xb = XBooster(model)           # 학습된 XGBoost/LightGBM 모형
scorecard = xb.to_scorecard()  # leaf weight → 전통 스코어카드 포인트로 변환
print(scorecard)
```

### 전통 스코어카드와의 성능 비교

```python
from sklearn.metrics import roc_auc_score

models = {
    'LR (WoE)': lr_model,
    '1-Depth GBM': depth1_model,
    'EBM': ebm_model,
    'Full GBM (depth=5)': full_model,
}

for name, model in models.items():
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    oot_auc = roc_auc_score(y_oot, model.predict_proba(X_oot)[:, 1])
    print(f"{name:20s} | Train AUC: {train_auc:.4f} | Valid AUC: {valid_auc:.4f} | OOT AUC: {oot_auc:.4f}")
```

---

## 5.5 실무 적용 시나리오

### 시나리오 1: 규제 모형 대체

전통 LR 스코어카드의 대체 모형으로 1-Depth GBM을 제안. GAM 구조이므로 변수별 효과를 점수표처럼 제출할 수 있다.

### 시나리오 2: 챌린저 모형

운영 중인 전통 모형(챔피언) 대비 1-Depth GBM(챌린저)의 성능 우위를 OOT에서 검증하여 교체 근거를 확보.

### 시나리오 3: 변수 탐색 도구

Full GBM을 바로 만들기 전에, 1-Depth GBM으로 **각 변수의 비선형 효과를 탐색**한다. WoE Classing의 구간 설정에도 참고할 수 있다 — "1-Depth GBM의 효과 곡선이 꺾이는 지점"이 자연스러운 Classing 경계가 된다.

### 시나리오 4: FICO 방식 — ML로 탐색, 전통으로 배포

Fahner (2018)의 transmutation 전략. Full GBM으로 패턴을 발견하고, 그 인사이트를 단조성 제약이 있는 스코어카드로 변환하여 배포한다.

---

## 5.6 참고 자료

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

## 5.7 요약

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

!!! tip "다음 섹션"
    [모델 검증](../part5_validation/index.md)에서 성능 지표, OOT 검증, 규제 프레임워크를 다룬다.
