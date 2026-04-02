---
title: "optbinning 실무 가이드"
---

# 부록 A: optbinning 실무 가이드

> Optimal Binning 라이브러리의 내부 동작 원리, 핵심 파라미터, WoE 변환, 다변량 처리까지 — 실무에서 알아야 할 모든 것

!!! note "대상 독자"
    이 부록은 [Coarse Classing](../part3_variable_selection/classing/coarse-classing.md)에서 소개한 optbinning의 기초 사용법을 넘어, **내부 최적화 원리와 실무 파라미터 튜닝**을 이해하고자 하는 실무자를 대상으로 합니다.

## 이 부록에서 다루는 내용

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [개요와 Pre-binning](optbinning-overview.md) | 라이브러리 아키텍처, 처리 파이프라인, Pre-binning 방식별 비교 |
| 2 | [최적화 엔진](optbinning-optimizer.md) | MIP/CP 문제 정식화, 제약조건, Solver 선택 |
| 3 | [핵심 파라미터 상세](optbinning-params.md) | 단조성, 샘플 크기, Bin 수, p-value, Special/Missing, 범주형, 정규화 |
| 4 | [Binning Table과 WoE 변환](optbinning-table-woe.md) | 결과 분석, 사후 검정, WoE/Event Rate/인덱스 변환 |
| 5 | [다변량 처리와 Scorecard](optbinning-pipeline.md) | BinningProcess 일괄 처리, Scorecard 클래스, 실무 체크리스트 |

---

<div class="source-ref" markdown>
**참고 문헌**

- Navas-Palencia, G. (2020). "Optimal Binning: Mathematical Programming Formulation." [arXiv:2001.08025](https://arxiv.org/abs/2001.08025)
- [optbinning 공식 문서](https://gnpalencia.org/optbinning/)
- [optbinning GitHub](https://github.com/guillermo-navas-palencia/optbinning)
</div>
