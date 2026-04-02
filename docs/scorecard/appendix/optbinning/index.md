---
title: "optbinning 실무 가이드"
---

# 부록 A: optbinning 실무 가이드

> Optimal Binning 라이브러리의 내부 동작 원리, 핵심 파라미터, WoE 변환, 다변량 처리까지 — 실무에서 알아야 할 모든 것

!!! note "이 부록의 위치"
    [Coarse Classing](../../part3_variable_selection/classing/coarse-classing.md)에서 소개한 optbinning의 기초 사용법을 넘어, **내부 최적화 원리와 실무 파라미터 튜닝**을 정리한다.

## 이 부록에서 다루는 내용

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [최적화 엔진](optimizer.md) | MIP/CP 문제 정식화, 제약조건, Solver 선택 |
| 2 | [핵심 파라미터 상세](params.md) | 단조성, 샘플 크기, Bin 수, p-value, Special/Missing, 범주형, 정규화 |
| 3 | [Binning Table과 WoE 변환](table-woe.md) | 결과 분석, 사후 검정, WoE/Event Rate/인덱스 변환 |
| 4 | [다변량 처리와 Scorecard](pipeline.md) | BinningProcess 일괄 처리, Scorecard 클래스, 실무 체크리스트 |

---

## 라이브러리 개요와 아키텍처

optbinning은 **수학적 최적화**(혼합정수계획법 MIP / 제약 프로그래밍 CP)로 binning 문제를 풀어, 단조성·최소 샘플·통계적 유의성 등 여러 제약을 **동시에** 만족하는 최적 구간을 찾는다.

### 처리 파이프라인

> **Pre-binning** → **최적화(MIP/CP)** → **Binning Table** → **WoE 변환** → **(선택) 스코어카드**

| 단계 | 역할 | 핵심 클래스 |
|------|------|-----------|
| Pre-binning | 초기 Fine Bin 생성 (CART, quantile 등) | `OptimalBinning` 내부 |
| 최적화 | 제약조건 하에서 IV 최대화하는 합병 조합 탐색 | `OptimalBinning` |
| Binning Table | 구간별 통계량 출력 (Count, WoE, IV 등) | `binning_table` |
| WoE 변환 | 원본 데이터를 WoE 값으로 치환 | `transform()` |
| 다변량 처리 | 전체 변수 일괄 binning + 변수 선별 | `BinningProcess` |
| 스코어카드 | 로지스틱 회귀 + 점수 변환 | `Scorecard` |

---

## Pre-binning — 초기 구간 생성

최적화 전에 연속형 변수를 **초기 Fine Bin**으로 나누는 단계다. 이 단계의 품질이 최종 결과에 직접 영향을 미친다.

| 방식 | `prebinning_method` | 원리 | 특징 |
|------|---------------------|------|------|
| **CART** (기본값) | `"cart"` | `DecisionTreeClassifier`로 정보 이득 기반 분할 | 타겟 반응률 차이가 큰 지점을 자동 탐지, 가장 적응적 |
| **MDLP** | `"mdlp"` | 최소 기술 길이 원리 기반 엔트로피 분할 | 이론적으로 정교하나 CART 대비 실무 차이 미미 |
| **Quantile** | `"quantile"` | 등빈도(equal-frequency) 분할 | 각 구간의 샘플 수 균등, 비지도 방식 |
| **Uniform** | `"uniform"` | 등간격(equal-width) 분할 | 값 범위 기준 균등 분할, 편향된 분포에 취약 |

```python
optb = OptimalBinning(
    name="utilization_rate",
    prebinning_method="cart",   # 기본값, 대부분 이대로 사용
    max_n_prebins=20,           # 초기 Fine Bin 최대 수 (기본 20)
    min_prebin_size=0.05,       # Fine Bin당 최소 샘플 비율
)
```

!!! tip "Pre-binning 후 정제"
    Pre-binning 결과에서 **순수 Bin**(Event=0 또는 Non-event=0)이 발생하면 WoE가 ±∞가 된다. optbinning은 이런 순수 Bin을 인접 Bin과 자동 합병하여 정제한 후 최적화 단계로 넘긴다.

---

<div class="source-ref" markdown>
**참고 문헌**

- Navas-Palencia, G. (2020). "Optimal Binning: Mathematical Programming Formulation." [arXiv:2001.08025](https://arxiv.org/abs/2001.08025)
- [optbinning 공식 문서](https://gnpalencia.org/optbinning/)
- [optbinning GitHub](https://github.com/guillermo-navas-palencia/optbinning)
</div>
