# 다변량 처리와 Scorecard

## 4.1 BinningProcess — 다변량 일괄 처리

실무에서는 수십~수백 개 변수를 동시에 binning해야 한다. `BinningProcess`가 이를 자동화한다.

```python
from optbinning import BinningProcess

variable_names = ["utilization_rate", "profit_margin", "sales_amount",
                  "years_in_business", "debt_ratio"]

binning_process = BinningProcess(
    variable_names=variable_names,
    categorical_variables=["industry_code"],  # 범주형 변수 지정

    # ── 전역 기본 파라미터 ──
    max_n_bins=7,
    min_bin_size=0.05,
    max_pvalue=0.05,

    # ── 변수별 개별 오버라이드 ──
    binning_fit_params={
        "utilization_rate": {"monotonic_trend": "descending"},
        "years_in_business": {"monotonic_trend": "ascending"},
        "profit_margin": {"monotonic_trend": "auto", "max_n_bins": 5},
    },

    # ── 자동 변수 선별 ──
    selection_criteria={"iv": {"min": 0.02}},  # IV < 0.02 변수 자동 제거

    # ── 병렬 처리 ──
    n_jobs=-1,  # 모든 CPU 코어 활용
)

binning_process.fit(X, y)
```

### 주요 파라미터

| 파라미터 | 설명 |
|---------|------|
| `variable_names` | 처리할 전체 변수 목록 |
| `categorical_variables` | 범주형으로 처리할 변수 목록 |
| `binning_fit_params` | 변수별 OptimalBinning 파라미터 오버라이드 (딕셔너리) |
| `selection_criteria` | IV 등 기준으로 변수 자동 선별. 예: `{"iv": {"min": 0.02}}` |
| `fixed_variables` | `selection_criteria`와 무관하게 반드시 포함할 변수 |
| `n_jobs` | 병렬 처리 코어 수 (`-1` = 전체) |

### 전체 변수 WoE 변환

```python
# 전체 DataFrame을 한 번에 WoE 변환
X_woe = binning_process.transform(X, metric="woe")

# 개별 변수 결과 확인
optb_util = binning_process.get_binned_variable("utilization_rate")
print(optb_util.binning_table.build())
```

### 변수 선별 결과 확인

```python
# 선별된 변수 목록
print(binning_process.get_support(names=True))

# 각 변수의 IV 확인
summary = binning_process.summary()
print(summary[["name", "dtype", "iv", "quality_score"]])
```

---

## 4.2 Scorecard 클래스

`BinningProcess` + 로지스틱 회귀 + 점수 변환을 하나의 파이프라인으로 통합한다.

```python
from optbinning import Scorecard
from sklearn.linear_model import LogisticRegression

scorecard = Scorecard(
    binning_process=binning_process,
    estimator=LogisticRegression(C=1.0, solver="lbfgs"),

    # ── 스코어 스케일링 ──
    scaling_method="pdo_odds",
    scaling_method_params={
        "pdo": 20,              # Points to Double the Odds
        "odds": 1/19,           # 기준 Odds (Bad:Good = 1:19)
        "scorecard_points": 600 # 기준 Odds에서의 점수
    },
    rounding=True,              # 정수 반올림
)

scorecard.fit(X_train, y_train)
```

### 스코어카드 테이블 출력

```python
# 요약 테이블
print(scorecard.table(style="summary"))

# 상세 테이블 (Bin별 점수)
print(scorecard.table(style="detailed"))
```

### 스케일링 방식

| `scaling_method` | 설명 |
|-----------------|------|
| `"pdo_odds"` | PDO(Points to Double the Odds) 기반. 업계 표준 |
| `"min_max"` | 점수 범위를 [min, max]로 고정. 예: `{"min": 300, "max": 850}` |

### 예측

```python
# 확률 예측
proba = scorecard.predict_proba(X_test)[:, 1]

# 점수 예측
scores = scorecard.score(X_test)

# 클래스 예측
labels = scorecard.predict(X_test)
```

---

## 4.3 실무 파라미터 체크리스트

| 상황 | 권장 설정 |
|------|----------|
| 첫 시도, 변수 특성 모름 | `monotonic_trend="auto"`, `solver="cp"`, `max_n_bins=7` |
| 업무 논리상 방향 확실 | `monotonic_trend="ascending"` 또는 `"descending"` 명시 |
| 소표본 (< 5,000건) | `min_bin_size=0.10`, `min_bin_n_event=20` |
| 대표본 (> 100,000건) | `max_n_prebins=50`으로 Fine Bin 늘려 정밀도 향상 |
| 0값 Mass Point 존재 | `special_codes=[0]`으로 0을 별도 Bin 분리 |
| 범주형 변수, 희귀 범주 多 | `cat_cutoff=0.05`로 5% 미만 범주 자동 그룹화 |
| 특정 Bin이 IV 독점 | `gamma=0.01~0.05`로 정규화 |
| 빠른 처리 필요 | `prebinning_method="quantile"`, `max_n_prebins=10` |
