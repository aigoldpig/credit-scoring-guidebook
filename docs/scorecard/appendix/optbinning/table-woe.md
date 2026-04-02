# Binning Table과 WoE 변환

## 3.1 Binning Table — 결과 분석

```python
optb.fit(X["utilization_rate"], y)
table = optb.binning_table.build()
print(table)
```

출력 예시:

```
  Bin              Count  Count (%)  Non-event  Event  Event rate     WoE       IV
  (-inf, 20.0)     1,850    18.5%      1,550    300     16.2%     -0.712   0.1153
  [20.0, 40.0)     2,100    21.0%      1,820    280     13.3%     -0.385   0.0342
  [40.0, 60.0)     2,350    23.5%      2,150    200      8.5%      0.098   0.0023
  [60.0, 80.0)     2,200    22.0%      2,100    100      4.5%      0.743   0.1050
  [80.0, inf)      1,500    15.0%      1,480     20      1.3%      1.452   0.2147
  Special              0     0.0%          0      0      0.0%        —        —
  Missing              0     0.0%          0      0      0.0%        —        —
  Totals          10,000   100.0%      9,100    900      9.0%        —    0.4715
```

### 사후 통계 검정

```python
optb.binning_table.analysis(pvalue_test="chi2")
```

| 검정 방식 | `pvalue_test` | 용도 |
|----------|---------------|------|
| 카이제곱 | `"chi2"` | 인접 Bin 간 Good/Bad 분포 차이 검정 (기본) |
| Fisher 정확 검정 | `"fisher"` | 소표본에서 카이제곱 대안 |

출력에는 인접 Bin 쌍별 p-value가 포함되어, **모든 인접 쌍이 p < 0.05인지** 한눈에 확인할 수 있다.

---

## 3.2 WoE 변환 — transform()

Binning이 완료되면 원본 데이터를 WoE 값으로 치환하여 로지스틱 회귀의 입력으로 사용한다.

```python
# WoE 변환 (기본)
X_woe = optb.transform(X["utilization_rate"], metric="woe")

# 다른 변환 옵션
X_event_rate = optb.transform(X["utilization_rate"], metric="event_rate")
X_indices    = optb.transform(X["utilization_rate"], metric="indices")
X_bins       = optb.transform(X["utilization_rate"], metric="bins")
```

| `metric` | 반환값 | 용도 |
|----------|-------|------|
| `"woe"` | 각 관측치의 WoE 값 | 로지스틱 회귀 입력 (기본) |
| `"event_rate"` | 소속 Bin의 Event Rate | EDA, 시각화 |
| `"indices"` | 0부터 시작하는 Bin 인덱스 | 프로그래밍 용도 |
| `"bins"` | Bin 라벨 문자열 (예: `"(-inf, 25.5]"`) | 리포팅 |

### Special/Missing 변환 제어

```python
X_woe = optb.transform(
    X["utilization_rate"],
    metric="woe",
    metric_special=0,    # Special Bin의 WoE (기본 0 = 중립)
    metric_missing=0,    # Missing Bin의 WoE (기본 0 = 중립)
)
```
