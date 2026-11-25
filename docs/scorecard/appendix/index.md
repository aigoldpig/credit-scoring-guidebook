---
title: 부록
---

# 부록

> 본 가이드북의 심화 주제와 실무 도구 가이드

!!! warning "부록 공통 부호 규약 — 먼저 읽을 것"
    부록 전체는 본문 Part 1~4와 같은 부호 체계를 쓴다.

    - **WoE 정의**: \(\text{WoE}_i = \ln(\%\text{Good}_i / \%\text{Bad}_i)\) → 우량 구간 WoE > 0, 불량 구간 WoE < 0
    - **모형 적합은 항상 Bad logit**: \(\ln\frac{p}{1-p} = \beta_0 + \sum_j \beta_j \text{WoE}_j\), \(p = P(\text{Bad})\), y=1=Bad
    - 따라서 WoE 변수의 **정상 부호는 \(\beta < 0\)** 이다. 단변량에서는 \(\hat\beta = -1\)이 정확히 성립하고, 다변량에서는 공선성 때문에 \(|\beta|\)가 1보다 작아지는 것이 일반적이다. **\(\beta\)가 양(+)으로 바뀌는 것이 부호 역전**이다.
    - 더미 방식에서 기준 구간을 **최위험 구간**으로 두면, 더 안전한 구간의 계수 \(\gamma\)는 **음수**(로짓 감소 = 더 안전)다.
    - 점수 변환(Part 5, Good Odds)은 계수를 다시 적합하지 않고 \(\ln(\text{Good Odds}) = -\text{logit}_{\text{Bad}}\)로 바꾼다. 구간 배점은 \(-\text{Factor}\cdot\beta_j\cdot\text{WoE}_{ij}\)(+ 상수 배분)이며, \(\text{Factor} = \text{PDO}/\ln 2\)다.
    - optbinning의 단조성 옵션은 **Event rate(불량률) 방향**이다. Event rate 증가 ⇔ 가이드북 WoE 감소.
    - 부록 C(LOWESS)는 로짓이 아니라 **단변량 부도확률 \(T(x)\)** 자체를 다룬다. \(T(x)\)가 클수록 위험하다.

## 부록 목차

| 섹션 | 제목 | 내용 |
|------|------|------|
| A | [optbinning 실무 가이드](optbinning/index.md) | Pre-binning 원리, MIP/CP 최적화 정식화, 핵심 옵션의 의미, Binning Table과 WoE 변환, 다변량 일괄 처리와 점수 변환 |
| B | [로지스틱 적합 비교](logistic-fitting/index.md) | WoE 직접 투입 vs 더미 변수 방식의 차이 · Stepwise 단계별 검정 원리 · Score / Wald / LR Test 해설 |
| C | [LOWESS 기반 미니모델링](lowess/index.md) | Moody's RiskCalc 계열의 LOWESS 기반 비모수적 재무비율 변환 기법 |
| D | [기호 용어집](glossary.md) | 본 가이드북에서 사용하는 기호와 용어 정리 |

## 읽는 순서

- **스코어카드 본문을 막 끝냈다면**: 부록 B(투입 방식과 검정) → 부록 A(구간화 도구) 순서를 권한다. 본문 Part 3~4에서 내린 결정의 "왜"를 보강한다.
- **기업 여신 모형을 다룬다면**: 부록 C를 먼저 읽는다. 소매 CSS의 WoE 변환과 같은 사상을 연속형 평활로 구현하는 방법이다.
- 기호가 헷갈리면 [기호 용어집](glossary.md)을 참조한다.
