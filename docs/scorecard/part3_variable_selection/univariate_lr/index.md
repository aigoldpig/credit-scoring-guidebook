# 단변량 로지스틱 회귀: 유의성 검정과 Classing 피드백 루프

> 변수 선정의 세 번째 단계 — WoE 계산 이후, 변수와 구간 경계의 통계적 유의성을 검정하고 Classing 품질을 진단하는 과정

!!! info "Odds 방향과 부호 — 이 섹션은 Bad Odds"
    단변량 로지스틱 회귀는 Bad를 \(y=1\)로 놓고 **Bad Odds** \(p/(1-p)\)의 로그(Bad logit)를 모델링한다. WoE = ln(%Good/%Bad)는 우량할수록 커지므로, WoE 변수의 정상 계수는 **음수**다(단변량에서는 정확히 −1).

    **Good Odds** \((1-p)/p\)로의 전환은 [스코어카드 변환](../../part5_scorecard/scorecard-and-rating.md)에서 이루어진다. 계수를 다시 적합하지 않고, \(\ln(\text{Good Odds}) = -\text{logit}_{\text{Bad}}\)로 부호만 뒤집어 점수에 반영한다.

**변수 선정**은 [Classing](../classing/index.md) → [WoE/IV](../woe_iv/index.md) → 단변량 로지스틱 회귀 → [정보영역별 변수 선정](../domain_selection/index.md)의 네 단계로 구성되며, 이 섹션은 그 세 번째 단계다.

WoE는 집계값이라 그 자체로는 표준오차도 p-value도 없다. 단변량 로지스틱 회귀를 거치면 **변수 전체가 유의한지(LRT)**, **인접 구간의 위험이 실제로 다른지(인접 Bin 차이 검정)** 를 통계적으로 판단할 수 있다.

!!! warning "먼저 알아둘 점 — 단변량 계수는 WoE에서 '계산'으로 나온다"
    절편 없는 One-Hot 회귀의 구간 계수는 \(\hat\beta_i = -\text{WoE}_i + \ln(N_B/N_G)\), WoE 치환 회귀의 계수는 \(\hat\beta = -1\)이 **정확히** 성립한다(닫힌 해). 그래서 단변량 단계에서 "β와 WoE가 맞는가"를 비교하는 것은 계산 검증일 뿐 Classing 진단이 아니다. 단변량 회귀의 실질적인 가치는 **표준오차를 이용한 검정**에 있고, β와 WoE의 관계가 진단 정보가 되는 것은 **다변량 모형과 검증(OOT) 표본**에서다.

> **Classing** → **WoE/IV 산출** → **단변량 로지스틱 회귀 (LRT · 인접 Bin 차이 검정)** → **Classing 피드백 루프** → **확정 or 재시도**

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [개념과 목적](concept.md) | WoE가 이미 계산되었는데 왜 회귀를 돌리는가? 무엇을 얻는가? |
| 2 | [One-Hot + No Intercept](one-hot-no-intercept.md) | k−1 더미, k개 더미 + 절편 제거, WoE 치환은 어떻게 다른가? |
| 3 | [β = −WoE + 상수 증명](beta-woe-proof.md) | 왜 계수가 WoE에서 정확히 계산되는가? 그 관계는 어디서 진단에 쓰이는가? |
| 4 | [유의성 검정](significance-test.md) | Wald·LRT로 무엇을 검정해야 하는가? 비유의 Bin은 어떻게 처리하는가? (기준 정본) |
| 5 | [Classing 피드백 루프](feedback-loop.md) | 구분되지 않는 Bin을 합병하고 재확인하는 절차는? |

!!! tip "다음 섹션"
    단변량 로지스틱 회귀를 통과한 변수는 [정보영역별 변수 선정](../domain_selection/index.md)에서 영역 내 다중공선성을 정리한 뒤, [다변량 모형(Part 4)](../../part4_modeling/index.md)으로 통합된다.
