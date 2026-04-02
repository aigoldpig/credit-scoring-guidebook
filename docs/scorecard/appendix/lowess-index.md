---
title: "LOWESS 기반 미니모델링과 기업 여신"
---

# 부록 C: LOWESS 기반 미니모델링 — 재무비율 변환과 기업 여신 신용평가

> Moody's RiskCalc 모델에서 활용되는 비모수적 재무비율 변환 기법의 방법론, 사상, 효과, 그리고 실무 적용 구조에 대한 기술 분석

!!! note "대상 독자"
    이 부록은 단변량 로지스틱 회귀 섹션의 **소매 여신 관점**을 보완하여, **기업 여신(corporate lending)** 영역에서의 미니모델링 방법론을 다룹니다. Moody's RiskCalc 모델 시리즈에서 공식 사용하는 LOWESS 기반 비모수적 변환 기법의 구조와 실무 적용을 정리합니다.

!!! tip "소매 vs 기업 미니모델링"
    본 가이드북의 단변량 로지스틱 회귀 섹션에서 다룬 방식은 **소매 CSS**에서의 WoE 기반 방식입니다. 이 부록에서 다루는 LOWESS 기반 미니모델링은 **Moody's RiskCalc**가 기업 여신 부도 예측에 사용하는 방식으로, 접근 철학은 같지만 기법이 다릅니다.

## 이 부록에서 다루는 내용

| 섹션 | 제목 | 내용 |
|------|------|------|
| 1 | [정의와 LOWESS 기초](lowess-intro.md) | 미니모델링 용어의 출처, LOWESS 알고리즘 구조, 핵심 파라미터 |
| 2 | [RiskCalc 프로세스와 사상](lowess-riskcalc.md) | 변환 절차, 백분위 변환, 비모수적 접근의 사상(비선형성·투명성·간결성) |
| 3 | [실증적 근거와 비율별 변환](lowess-effects.md) | 5가지 효과(비선형성·정규화·이상치·한계효과·강건성), 재무비율 범주별 변환 특성 |
| 4 | [3단계 모델 아키텍처](lowess-architecture.md) | Transform → Model → Map 구조, 프로빗 모형, 최종 매핑 |
| 5 | [실무 활용과 한계](lowess-application.md) | 기업 여신 6대 활용 영역, 국내 은행 접점, 한계와 고려사항 |

---

<div class="source-ref" markdown>
**참고 문헌**

- Falkenstein, E., Boral, A., & Carty, L. (2000). "RiskCalc for Private Companies: Moody's Default Model." Moody's Investors Service.
- Dwyer, D., Kocagil, A., & Stein, R. (2004). "The Moody's KMV EDF RiskCalc v3.1 Model." Moody's KMV.
- Moody's Analytics (2015). "RiskCalc 4.0 France." Modeling Methodology, Quantitative Research Group.
- Kocagil, A. et al. (2002). "Moody's RiskCalc Model for Privately-Held U.S. Banks." Moody's KMV.
- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots." JASA 74(368): 829-836.
</div>
