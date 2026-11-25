---
title: "① 모형 개발"
---

# ① 모형 개발 — 가이드북의 핵심 영역

본 가이드북의 전통 스코어카드(Part 1~5)와 머신러닝(Part 1~5)은
모형 개발의 전 과정을 다룬다. 이 페이지는 그 내용을 여신 프로세스의 관점에서 다시 묶어,
**모형 개발 단계가 무엇을 만들어 다음 단계로 넘기는지**를 정리한다.

## 가이드북 매핑

| 단계 | 내용 | 가이드북 위치 |
|---|---|---|
| 요건 정의 | 모집단·세그먼트, Good/Bad 정의, 관찰 시점·성과 기간·표본 기간, 거절추론 | [스코어카드 Part 1 — 요건 정의](../scorecard/part1_overview/requirements.md) |
| 변수 처리 | Classing, WoE/IV 변환, 단변량 LR, 정보영역별 대표 변수 선정 | [스코어카드 Part 3](../scorecard/part3_variable_selection/index.md) |
| 모형 학습 | 로지스틱 회귀(Full Model), GBM/RF 등 | [스코어카드 Part 4](../scorecard/part4_modeling/index.md), [ML Part 1](../ml/part1_overview/index.md)~[3](../ml/part3_tree_ensemble/index.md) |
| 해석·설명 | SHAP, EBM, fANOVA 등 | [ML Part 4](../ml/part4_interpretation/index.md) |
| 산출물 변환 | p̂ → Score(PDO 변환) → 등급 매핑 | [스코어카드 Part 5 — 스코어카드 변환 & 등급화](../scorecard/part5_scorecard/scorecard-and-rating.md) |
| 변별력 평가 | KS, AR | [스코어카드 Part 5 — 성능 평가](../scorecard/part5_scorecard/performance-ks-ar-gini.md), [ML Part 5](../ml/part5_validation/index.md) |
| 안정성 평가 | PSI, CSI, OOT 검증 | [OOT 검증](../scorecard/part5_scorecard/oot-validation.md), [모니터링](../scorecard/part5_scorecard/monitoring-and-operations.md) |
| 캘리브레이션 | 등급별 PD 산정, 예측 PD vs 실현 부도율(Back-testing) | [모니터링 — Back-testing](../scorecard/part5_scorecard/monitoring-and-operations.md), [PD 용어 정리](terms/pd.md) |

## 이 단계가 다음 단계로 넘기는 것

모형 개발의 산출물은 "모형 파일" 하나가 아니다. 심사·사후관리·리스크관리 부서가 실제로 쓰는 것은 다음 묶음이다.

| 산출물 | 내용 | 주로 쓰는 단계 |
|---|---|---|
| **평점표(스코어카드)** 또는 모형식 | 변수·구간별 배점, 또는 p̂ 산출식 | 심사 시스템 탑재(②) |
| **등급 체계** | 등급별 점수 구간, 구성비, 등급별 불량률 | 승인 cross-matrix, 한도 밴드(②③) |
| **등급별 PD** | 등급마다 붙이는 PD 숫자(관측 부도율 → 필요 시 CT 기준 레벨링) | 금리 EL, 충당금, 자본(②⑤) |
| **검증 결과** | 개발·OOT 표본의 KS·AR, PSI, 등급별 단조성 | 모형 승인, 정기 검증(⑤) |
| **모니터링 기준** | 지표별 관리 임계값, 재개발·리캘리브레이션 판단 기준 | 운영 중 모니터링(⑤) |
| **개발 문서** | 데이터·Bad 정의·변수 선정 근거·한계 | 내부 검증, 감독 대응(⑤) |

## 서열과 수준의 역할 분담

모형의 1차 목적은 **서열화(ordering)** — 누가 더 위험한지를 올바르게 줄 세우는 것이다.
그래서 모형 자체의 품질은 p̂의 절대값보다 **변별력(KS, AR)**으로 먼저 판단한다
([분류가 아닌 서열화](../scorecard/part1_overview/ranking-not-classification.md) 참고).

그러나 금리·충당금·자본처럼 PD가 **금액**으로 바뀌는 단계([p̂ / PD 활용 방식 종합](index.md))에서는
PD의 **수준**이 맞아야 한다. 이 역할은 보통 모형이 아니라 **캘리브레이션** 단계가 맡는다.

- **서열** — 모형(로지스틱 회귀 또는 ML)이 만든 점수의 순서
- **수준** — 등급별로 붙이는 PD 숫자. 개발 표본의 관측 부도율에서 출발해, 목적에 따라 장기 평균(CT) 기준으로 레벨링하거나 경기 상황을 반영해 조정한다

따라서 개발 단계에서는 변별력 검증과 별도로, **등급별 예측 PD와 실현 부도율이 맞는지**를 확인하는 캘리브레이션 검증까지 마쳐야 산출물이 완성된다. 다운샘플링으로 개발한 모형이라면 표본 불량률이 모집단과 다르므로 이 단계가 특히 중요하다.

## 용도별로 다른 모형

같은 방법론이라도 쓰임새에 따라 모집단·타겟·변수가 달라진다.

| 모형 | 적용 시점 | 쓰이는 단계 | 정의 |
|---|---|---|---|
| AS (Application Scoring) | 신청 시점 | ② 심사·승인 | [CSS란 — 목적 기준 분류](../scorecard/part1_overview/what-is-css.md) |
| BS (Behavior Scoring) | 대출 실행 후 주기적 | ③ 사후 관리 | 위와 같음 |
| Collection Scoring | 연체 발생 후 | ④ 부실·추심 | 위와 같음 |

하나의 기관이 상품·용도별로 여러 모형을 동시에 운용하는 것이 일반적이므로,
개발 착수 시 **이 모형이 어느 단계의 어떤 의사결정에 쓰이는지**를 먼저 확정해야
타겟 정의와 성과 기간, 검증 기준이 함께 정해진다.
