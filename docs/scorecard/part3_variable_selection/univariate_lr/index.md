# 단변량 로지스틱 회귀: 유의성 검정과 Classing 피드백 루프

> 변수 선정의 마지막 단계 — WoE 계산 이후, 각 구간의 통계적 유의성을 검정하고 Classing 품질을 진단하는 전 과정

!!! info "Odds 방향 — 이 섹션은 여전히 Bad Odds"
    단변량 로지스틱 회귀는 Bad를 \(y=1\)로 놓고 **Bad Odds** \(p/(1-p)\)를 모델링한다. [β ≈ WoE 증명](beta-woe-proof.md)의 수식도 Bad Odds 기반이다.

    **Good Odds** \((1-p)/p\)로의 전환은 [스코어카드 변환](../../part5_scorecard/scorecard-and-rating.md)에서 이루어진다. 점수 공식 \(\text{Score} = A - B \times \text{logit}\)의 마이너스 부호가 방향을 뒤집는 역할을 한다.

**변수 선정**은 [Classing](../classing/index.md) → [WoE/IV](../woe_iv/index.md) → 단변량 로지스틱 회귀의 세 단계로 구성되며, 이 섹션에서는 그 마지막 단계인 **단변량 로지스틱 회귀**를 다룬다.

WoE는 단순 집계값이라 표준오차도, p-value도 없다. 단변량 로지스틱 회귀를 통해 비로소 각 Bin에 대한 **통계적 유의성**이 산출되며, Classing의 품질을 객관적으로 검증할 수 있다.

> **Classing** → **WoE/IV 산출** → **단변량 로지스틱 회귀 (유의성 검정)** → **Classing 피드백 루프** → **확정 or 재시도**

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [개념과 목적](concept.md) | WoE가 이미 계산되었는데 왜 다시 회귀를 돌리는가? |
| 2 | [One-Hot + No Intercept](one-hot-no-intercept.md) | 왜 k-1 더미 대신 k개 더미 + 절편 제거를 사용하는가? |
| 3 | [β ≈ WoE 증명](beta-woe-proof.md) | No Intercept 회귀에서 추정 계수가 WoE와 일치하는 이유는? |
| 4 | [유의성 검정](significance-test.md) | Wald Test와 LRT로 Bin별·변수별 유의성을 어떻게 판단하는가? |
| 5 | [Classing 피드백 루프](feedback-loop.md) | 비유의 Bin을 어떻게 처리하며, 최종 확정까지의 절차는? |

!!! tip "다음 섹션"
    단변량 로지스틱 회귀를 마치면 확정된 변수들을 [모델링 · 스코어카드](../../part4_modeling/index.md)에서 다변량 모형으로 통합하고 점수로 변환한다.
