---
title: Classing
---

# Classing: 연속형 변수의 구간화

> 변수 선정의 첫 단계 — 연속형 변수를 의미 있는 구간으로 나누는 전 과정

**변수 선정**은 Classing → [WoE/IV](../woe_iv/index.md) → [단변량 로지스틱 회귀](../univariate_lr/index.md)의 세 단계로 구성되며, 이 섹션에서는 그 첫 단계인 **Classing**(구간화)을 다룬다. 연속형 변수를 구간화하여 각 구간에 독립적인 WoE를 부여하는 전처리 단계로, Classing이 필요한 이유와 방법론은 [왜 Classing인가](why-classing.md)에서 상세히 다룬다.

> **Fine Classing** (20~50개 초기 구간) → **WoE 패턴 확인** (단조성·변별력) → **Coarse Classing** (5~10개 최종 구간) → **단변량 로지스틱 회귀** (유의성 검정) → **확정 or 재시도** (피드백 루프)

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [왜 Classing인가](why-classing.md) | 연속형 변수를 그대로 쓸 수 없는 이유는? 어떤 문제를 해결하는가? |
| 2 | [Fine Classing](fine-classing.md) | 초기 구간화의 목적과 방법은? 결과에서 무엇을 봐야 하는가? |
| 3 | [Coarse Classing](coarse-classing.md) | 합병 알고리즘(단조성·ChiMerge·optbinning)과 기준 5가지는? |
| 4 | [의사결정 기준 종합](decision-criteria.md) | Fine → Coarse 전 과정의 체크리스트와 피드백 루프는? |

!!! tip "다음 섹션"
    Classing이 완료되면 각 Bin에 [WoE/IV](../woe_iv/index.md)를 부여하여 변수의 변별력을 정량 평가한다.
