---
title: WoE / IV
---

# WoE / IV: Weight of Evidence와 Information Value

> 연속형 X를 log-odds 공간으로 변환하는 원리와 변수 변별력 측정

!!! info "Odds 방향: Bad Odds"
    개요~WoE/IV 섹션까지는 **Bad Odds** \(p/(1-p)\)를 사용한다. 여기서 \(p\)는 부도(Bad) 확률이다. [스코어카드 변환](../../part5_scorecard/scorecard-and-rating.md) 단계에서 Good Odds \((1-p)/p\)로 전환된다.

[Classing](../classing/index.md)에서 연속형 변수를 구간(Bin)으로 분할하였다. 이제 각 Bin에 **WoE(Weight of Evidence)** 값을 부여하여 원래 X 공간을 log-odds 공간으로 변환하고, **IV(Information Value)**로 변수 전체의 변별력을 측정한다.

> **Classing** \(\xrightarrow{\text{Bin별 Good/Bad 집계}}\) **WoE 산출** \(\xrightarrow{\text{X → WoE 변환}}\) **IV 산출** \(\xrightarrow{\text{변수 선택}}\) **단변량 로지스틱 회귀 투입**

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [WoE: 정의·산출·해석](woe.md) | WoE는 어떻게 산출하며, 양수·음수는 무엇을 의미하는가? |
| 2 | [X → WoE 공간 변환](woe-transform.md) | 원래 X 공간과 WoE 공간은 어떻게 다르며, Dummy 대비 장점은? |
| 3 | [IV: 정의·산출·변수 선택](iv.md) | IV는 어떻게 산출하며, 변수 선택 기준은 무엇인가? |

WoE/IV로 변수를 선별한 뒤에는 [단변량 로지스틱 회귀](../univariate_lr/index.md)에서 각 Bin의 통계적 유의성을 검정하고 Classing 품질을 진단한다.
