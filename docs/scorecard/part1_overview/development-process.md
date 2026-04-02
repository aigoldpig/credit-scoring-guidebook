# 개발 파이프라인

## 7.1 전체 흐름

CSS 모형 개발은 크게 **요건 정의 → 데이터 준비 → 변수 선정 → 모델링 → 스코어링 → 검증**의 흐름으로 진행된다.

> **요건 정의** (모집단·Target·Segment) → **데이터 준비** (수집·추출·가공) → **변수 선정** (Classing·WoE/IV) → **모델링** (Logistic Regression) → **스코어링** (PDO·Scaling·Grading) → **검증** (PSI·AR·K-S)

아래에서 각 단계가 **무엇을 하는 것인지, 왜 필요한지, 어떤 산출물이 나오는지**를 살펴본다. 각 단계의 상세 방법론은 해당 Part에서 다룬다.

## 7.2 각 단계 Overview

**① 요건 정의** — [개요 (본 섹션)](index.md)

모형 개발의 출발점이다. **누구를 대상으로(모집단)**, **무엇을 예측할 것인지(Target)**, **어떤 기간을 볼 것인지(성과 기간)**를 확정한다. 필요 시 상품·고객군별로 세부 모형(Segment)을 분리한다.

> 산출물: Bad/Good/Indeterminate 정의서, 성과 기간 기준, 모형 Segment 설계

---

**② 데이터 준비**

정보 영역별(고객 인적사항, 여신·수신 실적, 카드 이용, CB 외부 정보 등)로 데이터를 수집·추출한다. 원천 데이터를 검증·정제한 뒤, 모형에 투입할 수 있는 형태의 **후보 변수(Feature)**를 생성한다. 이 단계에서 변수의 가감승제, 기간별 집계, 합성 변수 생성 등의 가공이 이루어진다.

> 산출물: 분석 항목 리스트 (수십~수백 개의 후보 변수)

---

**③ 변수 범주화(Classing)** — [Classing](../part3_variable_selection/classing/index.md)

연속형 변수를 구간으로 나누고(Fine Classing), 유사한 불량률 패턴의 구간을 합치며(Coarse Classing), 최종적으로 정수 값을 부여(Recoding)하는 과정이다. 이 단계를 통해 각 변수의 **불량률 패턴과 단변량 예측력**을 1차적으로 파악한다.

> 산출물: 변수별 Classing 테이블 (구간, 건수, 불량률, IV, K-S 등)

---

**④ 변수 변별력 평가(WoE / IV)** — [WoE/IV](../part3_variable_selection/woe_iv/index.md)

Classing된 각 구간에 **WoE(Weight of Evidence)** 값을 부여하여 변수를 변환한다. WoE는 각 구간이 Good과 Bad를 얼마나 잘 구분하는지를 나타내는 지표이며, 이를 구간별로 합산한 **IV(Information Value)**로 변수 전체의 예측력을 정량 평가한다. IV가 기준치 이하인 변수는 이 단계에서 탈락시킨다.

> 산출물: 변수별 WoE 테이블, IV 기반 변수 스크리닝 결과

---

**⑤ 모형 적합(Logistic Regression)** — [이론](../part2_theory/index.md) · [단변량 로지스틱 회귀](../part3_variable_selection/univariate_lr/index.md)

선별된 후보 변수들을 **로지스틱 회귀(Logistic Regression)**에 투입하여 최종 변수를 선택하고 계수(β)를 추정한다. 정보 영역별로 부분 회귀(Partial Logistic Regression)를 먼저 수행한 뒤, 전체 영역을 통합하여 최종 모형을 적합시킨다. 이 과정에서 **다중공선성 제거, 계수 부호 검정, 업무적 타당성 검토**가 병행된다.

> 산출물: 최종 변수 목록, 회귀계수(Estimate), 모형 적합 결과(Wald 검정, p-value 등)

---

**⑥ 평점화·등급화** — [모델링 · 스코어카드](../part4_modeling/index.md)

추정된 회귀계수를 **PDO(Point to Double Odds)** 체계를 이용해 점수(Score)로 변환하여 평점표(Scorecard)를 구성한다. 세부 모형이 여러 개인 경우 평점 통합(Scaling)을 수행하고, 최종적으로 점수를 **등급(Grade)**으로 구간화한다. 점수가 높을수록 우량하도록 설계하는 것이 일반적이다.

> 산출물: 평점표(항목별 배점), 등급 체계(등급별 평점 구간, 불량률, 구성비)

---

**⑦ 모형 검증** — [모델링 · 스코어카드](../part4_modeling/index.md)

개발된 모형이 실무에서 안정적으로 작동하는지를 검증한다. 크게 두 가지 관점에서 평가한다.

- **안정성 검증:** 개발 시점과 검증 시점 간 점수 분포 변화가 작은지 확인한다 (PSI, CAR 등).
- **변별력 검증:** 모형이 Good과 Bad를 얼마나 잘 구분하는지 평가한다 (AUROC, AR, K-S, IV 등). 등급별 불량률 서열화(SDR)가 유지되는지도 확인한다.

> 산출물: 모형 검증 보고서 (Front-end / Back-end Report)

!!! note "반복적 프로세스"
    위 흐름은 선형적으로 한 번에 끝나는 것이 아니다. 모형 적합 결과에 따라 변수 범주화를 다시 조정하거나, 검증 결과가 기준에 미달하면 변수 선택부터 재수행하는 등 **여러 차례의 반복(iteration)**이 일반적이다.

## 7.3 본 가이드북과의 매핑 요약

| 단계 | 가이드북 | 핵심 키워드 |
|------|------|------|
| ① 요건 정의 | **[개요](index.md)** | CSS 정의, 모형 분류, Target, Segment, Vintage, Roll Rate |
| ② 이론 | **[이론](../part2_theory/index.md)** | Logit, Sigmoid, MLE, Odds |
| ③ 변수 선정 — Classing | **[Classing](../part3_variable_selection/classing/index.md)** | Fine/Coarse Classing, Recoding |
| ④ 변수 선정 — WoE/IV | **[WoE/IV](../part3_variable_selection/woe_iv/index.md)** | Weight of Evidence, Information Value |
| ⑤ 변수 선정 — 단변량 로지스틱 회귀 | **[단변량 로지스틱 회귀](../part3_variable_selection/univariate_lr/index.md)** | Partial LR, 다중공선성, 유의성 검정 |
| ⑥ 모델링 · 스코어카드 | **[Full Model](../part4_modeling/index.md)** | 다변량 회귀, PDO, Scaling, Grading, PSI, AUROC |
| 부록 | **[부록](../appendix/index.md)** | WoE vs Dummy, Stepwise, 통계 검정 |

!!! success "학습 가이드"
    개요에서 전체 구조를 살펴본 뒤, 이론에서 수학적 기반을 학습하고, 변수 선정 → 모델링 → 스코어카드 순서로 실제 개발 절차를 따라가는 것을 권장한다. 각 섹션은 독립적으로 참조할 수 있도록 구성되어 있으나, 처음 학습하는 경우 순서대로 읽는 것이 효과적이다.
