# 개발 파이프라인

## 7.1 전체 흐름

CSS 모형 개발은 크게 **요건 정의 → 데이터 준비 → 변수 선정 → 모델링 → 스코어링 → 검증**의 흐름으로 진행된다.

> **① 요건 정의** (모집단·Target·Segment·기간) → **② 데이터 준비** (수집·추출·가공·표본 분할) → **③ Classing** → **④ WoE/IV** → **⑤ 모형 적합** (Logistic Regression) → **⑥ 평점화·등급화** (PDO·Scaling·Grading) → **⑦ 모형 검증** (KS·AR·PSI·캘리브레이션)

아래에서 각 단계가 **무엇을 하는 것인지, 왜 필요한지, 어떤 산출물이 나오는지**를 살펴본다. 각 단계의 상세 방법론은 해당 Part에서 다룬다.

## 7.2 각 단계 Overview

**① 요건 정의** — [요건 정의](requirements.md)

모형 개발의 출발점이다. **누구를 대상으로(모집단)**, **무엇을 예측할 것인지(Target)**, **어떤 기간을 볼 것인지(관찰 시점·성과 기간·표본 기간)**를 확정한다. 필요 시 상품·고객군별로 세부 모형(Segment)을 분리하고, AS모형이라면 거절 건을 어떻게 다룰지(거절추론)도 이 단계에서 정한다.

> 산출물: Bad/Good/Indeterminate 정의서, 기간 설정 기준, 모형 Segment 설계, 사전 제외 기준

---

**② 데이터 준비** — [변수 선정](x-variables.md)

정보 영역별(고객 인적사항, 여신·수신 실적, 카드 이용, CB 외부 정보 등)로 데이터를 수집·추출한다. 원천 데이터를 검증·정제한 뒤, 모형에 투입할 수 있는 형태의 **후보 변수(Feature)**를 생성한다. 이 단계에서 변수의 가감승제, 기간별 집계, 합성 변수 생성 등의 가공이 이루어진다. 관찰 시점 이후의 정보가 변수에 섞이지 않았는지(정보 누수) 반드시 확인한다.

!!! tip "개발 데이터는 끝까지 돌아도 틀릴 수 있다"
    데이터 결합·정제 코드는 결함이 있어도 끝까지 돌고 숫자가 나온다. 결합 전후 행수와 고유 키 수, 중복 제거의 정렬 기준이 라벨을 한쪽으로 치우치게 하지 않는지 확인한다. 최신 상태로 덮어쓰는 마스터 정보(현재 대표자, 현재 업종 등)를 과거 관찰 시점에 그대로 붙이지 않도록 주의한다.

개발 표본과 검증 표본도 이 단계에서 나눈다. 같은 기간 안에서 무작위로 떼어 둔 **검증 표본**과, 표본 기간 이후 시점의 **OOT(Out-of-Time) 표본**을 따로 확보해야 이후 단계에서 과적합과 시점 안정성을 평가할 수 있다([OOT 검증](../part5_scorecard/oot-validation.md)).

> 산출물: 분석 항목 리스트 (수십~수백 개의 후보 변수), 개발·검증·OOT 표본

---

**③ 변수 범주화(Classing)** — [Classing](../part3_variable_selection/classing/index.md)

연속형 변수를 구간으로 나누고(Fine Classing), 유사한 불량률 패턴의 구간을 합치며(Coarse Classing), 최종적으로 정수 값을 부여(Recoding)하는 과정이다. 이 단계를 통해 각 변수의 **불량률 패턴과 단변량 예측력**을 1차적으로 파악한다.

> 산출물: 변수별 Classing 테이블 (구간, 건수, 불량률, IV, KS 등)

---

**④ 변수 변별력 평가(WoE / IV)** — [WoE/IV](../part3_variable_selection/woe_iv/index.md)

Classing된 각 구간에 **WoE(Weight of Evidence)** 값을 부여하여 변수를 변환한다. 가이드북의 WoE는 \(\ln(\%\text{Good}/\%\text{Bad})\)로, 우량 구간일수록 크다. 이를 구간별로 합산한 **IV(Information Value)**로 변수 전체의 예측력을 정량 평가하며, IV가 기준치 이하인 변수는 이 단계에서 탈락시킨다.

> 산출물: 변수별 WoE 테이블, IV 기반 변수 스크리닝 결과

---

**⑤ 모형 적합(Logistic Regression)** — [이론](../part2_theory/index.md) · [단변량 로지스틱 회귀](../part3_variable_selection/univariate_lr/index.md) · [정보영역별 선정](../part3_variable_selection/domain_selection/index.md) · [다변량 회귀](../part4_modeling/multivariate-regression.md)

선별된 후보 변수들을 **로지스틱 회귀(Logistic Regression)**에 투입하여 최종 변수를 선택하고 계수(β)를 추정한다. 정보 영역별로 부분 회귀(Partial Logistic Regression)를 먼저 수행해 영역별 대표 변수를 정한 뒤, 전체 영역을 통합하여 최종 모형(Full Model)을 적합시킨다. 이 과정에서 **다중공선성 점검, 계수 부호 점검, 업무적 타당성 검토**가 병행된다. WoE로 투입한 변수의 정상 부호는 **β < 0**(우량 구간일수록 Bad 로그 오즈 감소)이며, 양(+)으로 바뀌면 공선성 등을 의심한다.

> 산출물: 최종 변수 목록, 회귀계수(Estimate), 모형 적합 결과(Wald 검정, p-value 등)

---

**⑥ 평점화·등급화** — [스코어카드 변환 & 등급화](../part5_scorecard/scorecard-and-rating.md)

추정된 회귀계수를 **PDO(Points to Double the Odds)** 체계를 이용해 점수(Score)로 변환하여 평점표(Scorecard)를 구성한다. 이때 계수를 다시 적합하지 않고 같은 β로 \(\ln(\text{Good Odds}) = -\,\text{logit}_{\text{Bad}}\)를 계산해, 점수가 높을수록 우량하도록 만든다. 세부 모형이 여러 개인 경우 평점 통합(Scaling)을 수행하고, 최종적으로 점수를 **등급(Grade)**으로 구간화한 뒤 등급별 PD를 산정한다.

> 산출물: 평점표(항목별 배점), 등급 체계(등급별 평점 구간, 불량률, 구성비), 등급별 PD

---

**⑦ 모형 검증** — [성능 평가](../part5_scorecard/performance-ks-ar-gini.md) · [OOT 검증](../part5_scorecard/oot-validation.md) · [모니터링](../part5_scorecard/monitoring-and-operations.md)

개발된 모형이 실무에서 안정적으로 작동하는지를 검증한다. 검증 보고서는 흔히 성과(y)가 필요 없는 **Front-end** 분석과, 성과가 관측된 뒤에야 가능한 **Back-end** 분석으로 나누어 구성한다.

- **Front-end (분포 안정성):** 개발 시점과 검증 시점 사이에 점수 분포와 변수별 분포가 얼마나 달라졌는지 확인한다. 점수 분포는 PSI, 변수별 분포는 CSI(Characteristic Stability Index)로 본다. 성과를 기다리지 않고 바로 계산할 수 있어 운영 초기부터 쓸 수 있다.
- **Back-end (성과 기반):** 성과 기간이 지난 표본으로 변별력(KS, AR), **등급별 불량률의 단조성**(등급이 내려갈수록 불량률이 높아지는지), 등급별 예측 PD와 실현 부도율의 일치(캘리브레이션)를 확인한다.

> 산출물: 모형 검증 보고서 (Front-end / Back-end 분석)

!!! note "반복적 프로세스"
    위 흐름은 선형적으로 한 번에 끝나는 것이 아니다. 모형 적합 결과에 따라 변수 범주화를 다시 조정하거나, 검증 결과가 기준에 미달하면 변수 선택부터 재수행하는 등 **여러 차례의 반복(iteration)**이 일반적이다.

## 7.3 본 가이드북과의 매핑 요약

| 단계 | 가이드북 | 핵심 키워드 |
|------|------|------|
| ① 요건 정의 | **[개요 — 요건 정의](requirements.md)** | 모집단, Target, Segment, 기간 설정, Vintage, Roll Rate, 거절추론 |
| ② 데이터 준비 | **[개요 — 변수 선정](x-variables.md)** | 변수풀, 대안정보, 사용 제한 변수, 표본 분할 |
| ③ Classing | **[Part 3 — Classing](../part3_variable_selection/classing/index.md)** | Fine/Coarse Classing, Recoding |
| ④ WoE/IV | **[Part 3 — WoE/IV](../part3_variable_selection/woe_iv/index.md)** | Weight of Evidence, Information Value |
| ⑤ 모형 적합 | **[Part 3 — 단변량 LR](../part3_variable_selection/univariate_lr/index.md)**, **[정보영역별 선정](../part3_variable_selection/domain_selection/index.md)**, **[Part 4 — 모델링](../part4_modeling/index.md)** | 단변량 검토, Partial LR, 다변량 회귀, 다중공선성, 부호 점검 |
| ⑥ 평점화·등급화 | **[Part 5 — 스코어카드 변환 & 등급화](../part5_scorecard/scorecard-and-rating.md)** | PDO, Scaling, Grading, 등급별 PD |
| ⑦ 모형 검증 | **[Part 5 — 성능 평가](../part5_scorecard/performance-ks-ar-gini.md)**, **[OOT 검증](../part5_scorecard/oot-validation.md)**, **[모니터링](../part5_scorecard/monitoring-and-operations.md)**, **[규제 프레임워크](../part5_scorecard/regulatory-framework.md)** | KS, AR, PSI, CSI, 캘리브레이션, 리캘리브레이션 |
| (사전 지식) 이론 | **[Part 2 — 이론](../part2_theory/index.md)** | Odds, Logit, Sigmoid, MLE |
| 부록 | **[부록](../appendix/index.md)** | optbinning 실무, WoE vs Dummy, Stepwise, Score·Wald·LR 검정, LOWESS, 기호 용어집 |

!!! success "학습 가이드"
    개요에서 전체 구조를 살펴본 뒤, 이론에서 수학적 기반을 학습하고, 변수 선정 → 모델링 → 스코어카드 순서로 실제 개발 절차를 따라가는 것을 권장한다. 각 섹션은 독립적으로 참조할 수 있도록 구성되어 있으나, 처음 학습하는 경우 순서대로 읽는 것이 효과적이다.
