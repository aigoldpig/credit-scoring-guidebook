# 모델링: 다변량 로지스틱 회귀

> 정보영역별 대표 변수를 통합해 최종 다변량 모형(Full Model)을 구축한다

[단변량 로지스틱 회귀(Simple LR)](../part3_variable_selection/univariate_lr/concept.md)는 변수 하나하나가 위험을 구분하는지 확인하는 단계였다. Full Model은 선별된 변수들을 **함께** 투입하여 최종 예측 모형을 완성하는 단계다.

> **① 전체 영역 통합 Full Model** →
> **② 회귀계수 검토 (부호·크기·유의성·VIF)** →
> **③ 최종 모형 확정** → [Part 5: 스코어카드 변환·성능 평가](../part5_scorecard/index.md)

!!! info "Odds 방향과 부호: Bad Odds"
    Part 4도 Part 3과 같이 y=1=Bad인 **Bad logit**으로 적합한다. WoE = ln(%Good/%Bad)는 우량할수록 크므로 정상 계수는 **음수**이고, 양수가 나오면 부호 역전이다. Good Odds로의 전환은 [Part 5](../part5_scorecard/scorecard-and-rating.md)에서 계수를 다시 적합하지 않고 부호만 뒤집어 이루어진다.

!!! note "정보영역별 변수 선정은 이전 단계에서 완료"
    동일 정보영역 내 Partial LR을 통한 대표 변수 확정은 [변수 선정](../part3_variable_selection/domain_selection/index.md) 단계에서 이미 수행되었다. 여기서는 영역별 대표 변수를 통합하여 Full Model을 구성하는 것부터 시작한다.

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [Simple LR vs Full Model](simple-vs-full.md) | 단변량과 다변량 모형은 무엇이 다른가? |
| 2 | [다변량 회귀](multivariate-regression.md) | Full Model을 어떻게 구성하고 계수를 검토하는가? |

!!! tip "다음 단계와 심화 학습"
    최종 모형이 확정되면 [Part 5](../part5_scorecard/index.md)에서 회귀계수를 점수로 변환하고(Anchor Score + PDO), 등급을 설계하고, KS·AR로 성능을 검증한다. WoE vs Dummy 투입 방식 비교, Stepwise 검정 이론, LOWESS 기반 기업 모형 등 심화 주제는 [부록](../appendix/index.md)에서 다룬다.
