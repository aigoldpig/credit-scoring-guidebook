# 모델링 · 스코어카드

> 다변량 모형 구축부터 스코어카드 변환, 등급화, 성능 평가까지

[단변량 로지스틱 회귀(Simple LR)](../part3_variable_selection/univariate_lr/concept.md)은 변수 하나하나의 단독 설명력을 확인하는 단계였다. Full Model은 선별된 변수들을 **함께** 투입하여 최종 예측 모형을 완성하고, 스코어카드로 변환한 후 성능을 검증하는 단계다.

> **① 전체 영역 통합 Full Model** →
> **② 회귀계수 검토 (부호·유의성·VIF)** →
> **③ 스코어카드 변환 (Anchor Score + PDO)** →
> **④ 성능 평가 (KS · AR · Gini)**

!!! note "정보영역별 변수 선정은 이전 단계에서 완료"
    동일 정보영역 내 Partial LR을 통한 대표 변수 확정은 [변수 선정](../part3_variable_selection/domain_selection/index.md) 단계에서 이미 수행되었다. 여기서는 영역별 대표 변수를 통합하여 Full Model을 구성하는 것부터 시작한다.

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [Simple LR vs Full Model](simple-vs-full.md) | 단변량과 다변량 모형은 무엇이 다른가? |
| 2 | [다변량 회귀](multivariate-regression.md) | Full Model을 어떻게 구성하고 계수를 검토하는가? |
| 3 | [스코어카드 변환 & 등급화](../part5_scorecard/scorecard-and-rating.md) | 회귀계수를 점수로 변환하고 등급을 설계하는 방법은? |
| 4 | [성능 평가 (KS·AR·Gini)](../part5_scorecard/performance-ks-ar-gini.md) | KS, AR, Gini, PSI로 모형을 어떻게 검증하는가? |

!!! tip "심화 학습"
    스코어카드 개발 전 과정을 마쳤다면, [부록](../appendix/index.md)에서 WoE vs Dummy 투입 방식 비교, Stepwise 검정 이론, LOWESS 기반 기업 모형 등 심화 주제를 학습할 수 있다.
