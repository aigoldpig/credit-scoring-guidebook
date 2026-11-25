---
title: "optbinning 실무 가이드"
---

# 부록 A: optbinning 실무 가이드

> 공개 라이브러리 optbinning이 "최적 구간화"를 어떤 원리로 푸는지 — Pre-binning, 수리 최적화, 결과 해석, 다변량 처리까지

!!! note "이 부록의 위치"
    [Coarse Classing](../../part3_variable_selection/classing/coarse-classing.md)에서 소개한 optbinning을 **내부 원리와 옵션의 의미** 중심으로 정리한다. 사용법(코드)이 아니라 "무엇을 최적화하고, 어떤 제약을 걸며, 결과를 어떻게 읽는가"를 다룬다. 구체적인 호출 방법은 [공식 문서](https://gnpalencia.org/optbinning/)를 참조한다.

!!! warning "방향 표기 주의"
    optbinning의 단조성 옵션은 **Event rate(불량률)** 의 방향을 뜻한다. 이 가이드북의 WoE(\(\ln(\%\text{Good}/\%\text{Bad})\))는 Event rate와 반대로 움직이므로, "Event rate 증가(ascending)" = "WoE 감소"다. 라이브러리가 출력하는 WoE도 Event rate와 반대 방향으로 움직인다(공식 튜토리얼: "a monotonically ascending event rate ensures a monotonically descending WoE").

## 이 부록에서 다루는 내용

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [최적화 엔진](optimizer.md) | 구간 병합을 이진 최적화로 정식화, 제약식, 소형 예시, 전역 최적의 의미, Solver |
| 2 | [핵심 옵션의 의미](params.md) | 단조성, 표본 크기, Bin 수, p-value, Special/Missing, 범주형, 정규화 |
| 3 | [Binning Table과 WoE 변환](table-woe.md) | 결과표 읽는 법(검산 예시), 사후 검정, 변환 방식 선택 |
| 4 | [다변량 처리와 점수 변환](pipeline.md) | 변수 일괄 처리, 자동 선별의 한계, 점수 변환 시 부호·규제 함정, 실무 체크리스트 |

---

## 라이브러리 개요와 처리 흐름

optbinning은 구간화를 **수학적 최적화 문제**(혼합정수계획법 MIP 또는 제약 프로그래밍 CP)로 풀어, 단조성·최소 표본·통계적 유의성 같은 여러 제약을 **동시에** 만족하면서 변별력(기본값: IV)이 가장 큰 구간 조합을 찾는다. 사람이 Fine Bin을 눈으로 보며 하나씩 합치는 전통적 Coarse Classing을 "제약 하 최적화"로 자동화한 것이다.

> **Pre-binning** → **최적화(MIP/CP)** → **Binning Table** → **변환(WoE 등)** → **(선택) 다변량 처리·점수 변환**

| 단계 | 역할 | 사람이 결정할 것 |
|------|------|-----------|
| Pre-binning | 연속 변수를 초기 Fine Bin으로 분할 | 분할 방식, 최대 Fine Bin 수, Fine Bin 최소 비중 |
| 최적화 | 제약을 만족하는 병합 조합 중 목적함수(IV) 최대인 것 선택 | 단조성 방향, 최소 표본, Bin 수, 유의성 기준 |
| Binning Table | 구간별 건수·불량률·WoE·IV 출력 | 결과가 업무 논리와 맞는지 검토 |
| 변환 | 원본 값을 WoE(또는 불량률·구간 번호)로 치환 | Special/Missing 구간의 값 처리 |
| 다변량 처리 | 여러 변수를 같은 규칙으로 일괄 구간화·선별 | 변수별 예외 설정, 선별 기준 |

---

## Pre-binning — 초기 구간 생성

최적화는 Pre-binning이 만든 Fine Bin 경계 **안에서만** 병합 조합을 고른다. 즉 Pre-binning에 없는 경계는 최종 결과에도 나타날 수 없으므로, 이 단계의 품질이 최종 결과의 상한을 정한다.

| 방식 | 원리 | 특징 |
|------|------|------|
| **CART** (기본값) | 의사결정나무의 불순도 감소가 큰 지점에서 분할 | 타겟 반응률 차이가 큰 지점을 자동 탐지. 가장 적응적 |
| **MDLP** | 최소 기술 길이(MDL) 원리에 따른 엔트로피 기반 분할 | 분할 중단 기준이 이론적으로 정의됨 |
| **Quantile** | 등빈도(equal-frequency) 분할 | 각 구간의 표본 수 균등, 타겟을 보지 않는 비지도 방식 |
| **Uniform** | 등간격(equal-width) 분할 | 값 범위 기준 균등 분할. 치우친 분포에서는 빈 구간이 생기기 쉬움 |

라이브러리 기본값은 CART 방식, 최대 Fine Bin 수 20개, Fine Bin당 최소 비중 5%다([공식 문서](https://gnpalencia.org/optbinning/binning_binary.html)). 최소 비중 5%와 최대 20개는 같은 의미의 제약이다(5% × 20 = 100%). 본문 [구간화 기준](../../part3_variable_selection/classing/decision-criteria.md)의 Fine Classing 권고(20~50개 구간)보다 거친 출발점이므로, 대표본에서는 Fine Bin 수를 늘리고 최소 비중을 낮추는 것을 검토한다.

!!! tip "순수 Bin이 생기면"
    Fine Bin 중 Bad가 0건이거나 Good이 0건인 구간(순수 Bin)은 WoE가 \(\pm\infty\)가 된다. 이런 구간은 최적화 전에 인접 구간과 합쳐 정리해야 한다. 결과표에서 극단적인 WoE가 보이면 Pre-binning 단계의 순수 Bin 처리를 먼저 의심한다.

---

<div class="source-ref" markdown>
**참고 문헌**

- Navas-Palencia, G. (2020). "Optimal Binning: Mathematical Programming Formulation." [arXiv:2001.08025](https://arxiv.org/abs/2001.08025)
- [optbinning 공식 문서](https://gnpalencia.org/optbinning/)
- [optbinning GitHub](https://github.com/guillermo-navas-palencia/optbinning)
</div>
