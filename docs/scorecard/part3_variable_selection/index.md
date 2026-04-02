---
title: "변수 선정"
---

# 변수 선정: Classing → WoE/IV → 단변량 회귀 → 정보영역별 선정

> 후보 변수를 구간화하고, 변별력을 측정하고, 통계적으로 검증한 뒤, 정보영역별로 대표 변수를 확정하는 전 과정

신용평가모형의 **변수 선정(Feature Selection)**은 네 단계로 구성된다. 각 단계는 독립적이면서도 순환적인 피드백 루프로 연결된다.

> **Classing** (구간화) → **WoE/IV** (변별력 측정) → **단변량 로지스틱 회귀** (유의성 검정) → **정보영역별 변수 선정** (영역 내 다중공선성 제거) → [Full Model](../part4_modeling/multivariate-regression.md)

## 네 단계 개요

| 단계 | 섹션 | 핵심 질문 |
|------|------|-----------|
| 1 | [Classing](classing/index.md) | 연속형 변수를 어떻게 구간화하는가? Fine → Coarse 과정은? |
| 2 | [WoE / IV](woe_iv/index.md) | 각 구간에 WoE를 부여하고, IV로 변수 전체의 변별력을 어떻게 측정하는가? |
| 3 | [단변량 로지스틱 회귀](univariate_lr/index.md) | WoE 산출 이후, 통계적 유의성을 어떻게 검정하는가? |
| 4 | [정보영역별 변수 선정](domain_selection/index.md) | 동일 정보영역 내 다중공선성을 제거하고, 영역별 대표 변수를 어떻게 확정하는가? |

!!! info "Odds 방향: Bad Odds"
    변수 선정 전 과정(Classing ~ 단변량 회귀)은 **Bad Odds** \(p/(1-p)\)를 사용한다. **Good Odds** \((1-p)/p\)로의 전환은 [스코어카드 변환](../part5_scorecard/scorecard-and-rating.md)에서 이루어진다.
