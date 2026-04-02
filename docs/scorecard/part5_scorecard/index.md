# 스코어카드 변환 · 성능 평가

> 회귀계수를 점수로 변환하고, 등급 체계를 설계하고, 판별력을 검증한다

[다변량 회귀](../part4_modeling/multivariate-regression.md)로 최종 모형이 확정되면, 회귀계수를 **스코어카드 점수**로 변환하고 **등급 체계**를 설계한 후 **판별력 지표**로 모형을 검증한다.

> **① 스코어카드 변환 (Anchor Score + PDO)** →
> **② 등급 설계 (Rating Grade)** →
> **③ 성능 평가 (KS · AR · Gini · PSI)** →
> **④ OOT 검증** →
> **⑤ 모니터링과 운영 (CSI · Back-testing · 리캘리브레이션)** →
> **⑥ 규제 프레임워크**

## 이 섹션에서 다루는 내용

| 섹션 | 제목 | 핵심 질문 |
|------|------|-----------|
| 1 | [스코어카드 변환 & 등급화](scorecard-and-rating.md) | 회귀계수를 점수로 변환하고 등급을 설계하는 방법은? |
| 2 | [성능 평가 (KS·AR·Gini)](performance-ks-ar-gini.md) | KS, AR, Gini, PSI로 모형을 어떻게 검증하는가? |
| 3 | [OOT 검증](oot-validation.md) | 미래 데이터에서도 모형이 유효한지 어떻게 확인하는가? |
| 4 | [모형 모니터링과 운영](monitoring-and-operations.md) | 배포 후 성능 열화를 어떻게 감지하고 대응하는가? |
| 5 | [규제 프레임워크](regulatory-framework.md) | SR 11-7, Basel IRB, 금감원 가이드라인의 요구사항은? |

!!! info "Odds 방향: Good Odds"
    스코어카드 변환부터는 **Good Odds** \((1-p)/p\)를 사용한다. 점수가 높을수록 우량 고객임을 의미하도록 설계하기 위함이다.
