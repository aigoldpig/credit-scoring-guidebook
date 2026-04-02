# OOT (Out-of-Time) 검증

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0" markdown>
<div style="background:var(--md-code-bg-color,#f5f5f5);border-radius:16px;padding:24px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)" markdown>
<div style="font-size:2rem;font-weight:700;color:#0064ff">OOT</div>
<div style="font-size:.85rem;color:#6b7280;margin:4px 0 8px">Out-of-Time</div>
<div style="font-size:.8rem;text-align:left;color:var(--md-default-fg-color--light)">개발 기간 이후의 데이터로 모형 성능을 검증한다.</div>
</div>
<div style="background:var(--md-code-bg-color,#f5f5f5);border-radius:16px;padding:24px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)" markdown>
<div style="font-size:2rem;font-weight:700;color:#0064ff">OOS</div>
<div style="font-size:.85rem;color:#6b7280;margin:4px 0 8px">Out-of-Sample</div>
<div style="font-size:.8rem;text-align:left;color:var(--md-default-fg-color--light)">같은 기간 내 Hold-out 샘플로 과적합을 점검한다.</div>
</div>
<div style="background:var(--md-code-bg-color,#f5f5f5);border-radius:16px;padding:24px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)" markdown>
<div style="font-size:2rem;font-weight:700;color:#0064ff">안정성</div>
<div style="font-size:.85rem;color:#6b7280;margin:4px 0 8px">Stability</div>
<div style="font-size:.8rem;text-align:left;color:var(--md-default-fg-color--light)">성능·변수 기여도·등급별 부도율이 시간에 걸쳐 유지되는가.</div>
</div>
</div>

모형 성능 평가에서 KS·AR·Gini가 아무리 높아도, **개발 데이터에서만 잘 작동하는 모형은 실전에서 쓸 수 없다.** OOT 검증은 모형이 **미래 데이터에서도 유효한지** 확인하는 가장 중요한 관문이다.

---

## 3.1 왜 OOT가 필요한가

### 개발 샘플의 한계

개발 샘플(Training)에서 산출한 KS·AR은 모형이 이미 "본 적 있는" 데이터에 대한 성능이다. 이 수치가 높다고 해서 미래에도 같은 성능이 보장되지 않는다.

| 검증 유형 | 데이터 원천 | 확인 사항 |
|-----------|-----------|-----------|
| **In-Sample (Training)** | 모형 학습에 사용한 데이터 | 모형이 데이터를 잘 학습했는가 (과적합 가능) |
| **OOS (Hold-out)** | 같은 기간 내 무작위 분리 | 과적합 여부 1차 점검 |
| **OOT (Out-of-Time)** | 개발 기간 **이후** 시점의 데이터 | 시간 변화에 대한 강건성 (실전 유효성) |

!!! warning "OOS만으로는 부족한 이유"
    OOS는 개발 샘플과 **같은 시점**의 데이터를 무작위 분리한 것이다. 같은 경제 환경, 같은 심사 정책, 같은 포트폴리오 구성을 공유하므로 **시간 변화에 따른 열화를 감지하지 못한다.** OOT만이 "이 모형이 내일도 통하는가"에 답할 수 있다.

### 개발·검증 샘플 설계

> **관측 윈도우** (변수 산출 기간) → **성과 윈도우** (불량 판정 기간)

| 구분 | 기간 (예시) | 용도 |
|------|------------|------|
| **개발 샘플** | 2021.01 ~ 2022.06 | 모형 학습 |
| **OOS** | 2021.01 ~ 2022.06 중 20~30% 무작위 분리 | 과적합 점검 |
| **OOT** | 2022.07 ~ 2023.06 | 시간 안정성 검증 |

!!! tip "OOT 기간 설정 실무"
    - OOT 기간은 최소 **6개월~1년** 이상이 권장된다.
    - **성과 윈도우(Performance Window)**가 12개월이면, OOT 관측 시점은 그만큼 앞서야 부도 레이블이 확정된다.
    - 경기 변동을 포착하려면 OOT 기간이 개발 기간과 **다른 경기 국면**을 포함하는 것이 이상적이다.

---

## 3.2 OOT에서 확인할 사항

### 변별력 안정성

| 항목 | 기준 | 판정 예시 |
|------|------|-----------|
| **KS 하락폭** | 개발 대비 5~10%p 이내 | 개발 KS 38 → OOT KS 33 (허용) |
| **AR/Gini 하락폭** | 개발 대비 0.05 이내 | 개발 AR 0.54 → OOT AR 0.50 (허용) |
| **AUC 하락폭** | 개발 대비 0.03 이내 | 개발 AUC 0.77 → OOT AUC 0.74 (허용) |

!!! warning "하락폭 기준은 절대적이지 않다"
    위 기준은 업계 관행이지, 규제가 정한 일률적 수치는 아니다. 모형 용도(심사 자동화 vs 참고용), 포트폴리오 특성(소매 vs 기업), 불량률 수준에 따라 허용 범위가 달라질 수 있다. 중요한 것은 **하락의 크기와 원인을 함께 보는 것**이다.

### 등급별 부도율 단조성 (Monotonicity)

OOT에서도 **등급이 나쁠수록 부도율이 높아지는** 단조 관계가 유지되어야 한다. 개발 샘플에서 단조였더라도 OOT에서 역전이 발생하면 해당 구간의 WoE 패턴이 변질되었을 가능성이 있다.

| 등급 | 개발 부도율 | OOT 부도율 | 판정 |
|------|-----------|-----------|------|
| 1 (최우량) | 0.3% | 0.4% | 정상 |
| 2 | 0.8% | 1.0% | 정상 |
| 3 | 1.5% | 1.8% | 정상 |
| 4 | 3.2% | 3.5% | 정상 |
| **5** | **5.5%** | **4.8%** | **역전 — 4등급보다 낮음, 점검 필요** |
| 6 (최불량) | 10.2% | 12.1% | 정상 |

!!! tip "단조성 위반 시 조치"
    ① 해당 등급의 관측 수가 극소수인지 확인 (소표본 변동 가능성)
    ② 해당 구간의 WoE 패턴이 개발 시점 대비 변했는지 CSI로 점검
    ③ 역전이 통계적으로 유의하면 해당 등급 구간 합병(Merge) 또는 Coarse Classing 재검토

### 점수 분포 안정성

OOT 샘플의 점수 분포가 개발 샘플과 유사한지 PSI로 확인한다. PSI 산출 방법과 임계값은 [성능 평가](performance-ks-ar-gini.md)의 PSI 섹션에서 다루었다.

| PSI | 판정 |
|-----|------|
| < 0.10 | 안정 — 점수 분포 유사 |
| 0.10 ~ 0.25 | 소폭 이동 — 원인 분석 필요 |
| > 0.25 | 유의미한 이동 — 모형 유효성 재검토 |

---

## 3.3 OOT 성능이 크게 떨어졌을 때

OOT에서 성능이 급락했다면, **모형의 문제인지 환경의 문제인지** 구분하는 것이 첫 번째 단계다.

### 원인 진단 체크리스트

| 점검 항목 | 모형 문제 시그널 | 환경 변화 시그널 |
|-----------|----------------|----------------|
| **PSI** | 낮음 (분포 유사한데 성능만 하락) | 높음 (모집단 자체가 변동) |
| **CSI** | 특정 변수 급등 → 해당 변수 데이터 이상 | 다수 변수 전반적 상승 → 경기·정책 변화 |
| **과적합** | OOS도 개발 대비 큰 차이 | OOS는 양호, OOT만 하락 |
| **데이터 누수** | 특정 변수 제거 시 성능 정상화 | 변수 제거해도 변화 없음 |

!!! danger "Target Leakage 점검"
    OOT 성능이 개발 대비 **과도하게 좋거나** 과도하게 나쁜 경우 모두 Target Leakage를 의심한다.

    - **과도하게 좋은 경우:** 불량 판정 이후에야 관측 가능한 변수(연체일수, 채권회수 단계 등)가 학습 변수에 포함
    - **과도하게 나쁜 경우:** 개발 데이터에 Leakage가 있어 성능이 부풀려졌고, OOT에서 Leakage가 해소되면서 "실제 성능"이 드러남

### 대응 방향

```
OOT 성능 급락
    │
    ├─ 모집단 변화 (PSI 높음)
    │   ├─ 일시적 (경기 충격 등) → 모니터링 지속, 다음 OOT 재확인
    │   └─ 구조적 (채널·상품·정책 변경) → 리캘리브레이션 또는 재개발
    │
    ├─ 과적합 (OOS도 하락)
    │   └─ 변수 축소, 정규화 강화, 데이터 증량 검토
    │
    └─ 데이터 누수 (Leakage)
        └─ 해당 변수 제거 후 재개발
```

---

## 3.4 실무 보고 — OOT 검증 보고서 구성

OOT 검증 결과는 모형 승인의 핵심 근거로, 내부 검증 부서와 감독당국에 제출하는 **공식 문서**다.

### 보고서 필수 항목

| 항목 | 내용 |
|------|------|
| **개발·OOT 샘플 정의** | 기간, 건수, 불량률, 제외 기준 |
| **변별력 비교** | KS, AR, AUC — 개발 vs OOT 대비표 |
| **PSI** | 점수 분포 이동량 |
| **등급별 부도율** | 개발 vs OOT 대비, 단조성 확인 |
| **변수 안정성** | 주요 변수 CSI |
| **종합 판정** | 모형 승인 / 조건부 승인 / 반려 |

!!! example "CB사 참고 — OOT 검증 관행"
    **NICE평가정보**와 **KCB**는 모형 개발 시 최소 1개 이상의 OOT 기간을 설정하여 검증한다. 개인 CSS의 경우 개발 기간 2~3년, OOT 기간 1년이 일반적이며, 기업 CSS는 불량 건수 확보를 위해 개발 기간을 더 길게 설정하는 경우가 많다.

    금감원 검사 시에도 **OOT 검증 결과 미비**는 주요 지적 사항 중 하나다. OOT 없이 배포된 모형은 검증 미흡으로 판정될 수 있다.

    <div class="source-ref">출처: 금융감독원 '신용위험 내부등급법 검증 실무 안내'(2019)</div>

!!! info "다음 단계"
    OOT 검증을 통과한 모형은 배포 후 [모니터링과 운영](monitoring-and-operations.md) 체계에 편입된다. PSI·CSI로 분포 안정성을, KS·AR 추이로 변별력을, Back-testing으로 예측 정확성을 지속 감시한다.
