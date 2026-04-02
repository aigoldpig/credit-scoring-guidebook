# 핵심 파라미터 상세

## 2.1 monotonic_trend — 단조성 제약

가장 중요한 파라미터다. WoE(또는 Event Rate)의 방향을 제약한다.

| 값 | 의미 | 사용 시점 |
|----|------|----------|
| `"auto"` | ML 분류기로 자동 탐지 (기본값) | 변수 특성을 모를 때 |
| `"ascending"` | WoE 단조증가 강제 | 사용률↑ → 위험↑ 등 |
| `"descending"` | WoE 단조감소 강제 | 소득↑ → 위험↓ 등 |
| `"peak"` | 역U형 (중간이 최대) | 업력, 특정 재무비율 |
| `"valley"` | U형 (중간이 최소) | 드문 경우 |
| `"auto_asc_desc"` | ascending/descending 중 자동 선택 | peak/valley 배제하고 싶을 때 |

!!! tip "`auto` 내부 동작"
    `"auto"` 설정 시 optbinning은 Pre-binning 결과의 Event Rate 패턴에서 **16개 특성**(추세 변화 횟수, 회귀 기울기 부호, 극값 위치, 볼록 껍질 비율 등)을 추출하고, 사전 학습된 결정 트리 분류기로 ascending / descending / peak / valley 중 하나를 예측한다. 이후 네 방향 각각으로 최적화를 실행해 **IV가 가장 높은 방향**을 최종 선택한다.

---

## 2.2 샘플 크기 제약

```python
optb = OptimalBinning(
    min_bin_size=0.05,        # Bin당 최소 전체 샘플의 5%
    max_bin_size=None,        # 최대 비율 (None=제한 없음)
    min_bin_n_event=10,       # Bin당 최소 Bad 건수
    min_bin_n_nonevent=10,    # Bin당 최소 Good 건수
)
```

---

## 2.3 Bin 수 제약

```python
optb = OptimalBinning(
    min_n_bins=3,             # 최소 Bin 수
    max_n_bins=7,             # 최대 Bin 수
)
```

!!! note "미지정 시"
    `max_n_bins=None`이면 solver가 IV를 최대화하는 Bin 수를 자동 결정한다. 실무에서는 **5~7**로 상한을 두는 것이 해석 가능성 측면에서 권장된다.

---

## 2.4 p-value 제약 — 인접 Bin 간 유의성

```python
optb = OptimalBinning(
    max_pvalue=0.05,                  # 인접 Bin 간 최대 p-value
    max_pvalue_policy="consecutive",  # "consecutive" 또는 "all"
)
```

| 파라미터 | 설명 |
|---------|------|
| `max_pvalue` | 인접 Bin 쌍의 Event Rate 차이에 대한 Z-test p-value 상한. 이를 초과하는 쌍은 합병 대상 |
| `max_pvalue_policy` | `"consecutive"`: 바로 인접한 쌍만 검정 (기본값). `"all"`: 모든 Bin 쌍 검정 |

!!! warning "p-value 산출 방식"
    최적화 **내부**에서는 Event Rate 차이에 대한 **Z-test**를 사용한다(연속형 타겟은 T-test). 이는 ChiMerge의 카이제곱 검정과 원리는 유사하지만 구현이 다르다. 사후 검증 시 `binning_table.analysis()`에서 Chi-square 또는 Fisher exact test를 별도로 수행할 수 있다.

---

## 2.5 Special Codes와 Missing 처리

```python
optb = OptimalBinning(
    special_codes=[-9, -99],          # 센티널 값 → 별도 Bin으로 분리
    # 또는 딕셔너리로 그룹화:
    # special_codes={"sentinel": [-9, -99], "zero": [0]},
)
```

- `special_codes`에 지정된 값은 최적화에서 제외되고, **별도의 Special Bin**으로 처리된다.
- **Missing 값**(NaN)도 자동으로 별도 Bin으로 분리된다.
- WoE 변환 시 `metric_special=0`, `metric_missing=0`이 기본값 (WoE=0, 즉 중립).

---

## 2.6 범주형 변수 처리

```python
optb = OptimalBinning(
    name="industry_code",
    dtype="categorical",        # 범주형 지정
    cat_cutoff=0.05,            # 빈도 5% 미만 범주는 자동 그룹화
)
```

---

## 2.7 정규화 — gamma 파라미터

```python
optb = OptimalBinning(
    gamma=0.01,   # L1 정규화 강도 (기본값 0)
)
```

gamma > 0이면 목적함수에서 **Bin 크기 불균형에 대한 페널티**가 추가된다. 특정 Bin이 IV를 독점하는 것을 방지하여 더 균형 잡힌 Bin 구성을 유도한다.
