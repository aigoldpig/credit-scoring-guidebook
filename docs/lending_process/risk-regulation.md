---
title: "⑤ 리스크·규제"
---

# ⑤ 리스크 관리·규제 — PD가 '금액'이 되는 영역

앞선 단계들이 개별 차주 수준의 의사결정이라면,
이 영역은 **포트폴리오 전체**를 대상으로 하며,
PD가 직접 **금액(원)**으로 환산되는 단계이다.
그래서 이 영역에서는 서열뿐 아니라 PD의 **수준(캘리브레이션)**이 맞아야 한다.

## IFRS 9 기대신용손실 (ECL)

IFRS 9(국내 K-IFRS 제1109호)는 2018년 1월 1일 이후 시작하는 회계연도부터 적용된 금융상품 회계기준으로,
대출 자산에 대해 **미래 예상 손실**을 선제적으로 충당금으로 적립하도록 한다.
그 이전 기준(IAS 39)은 손실 사건이 발생해야 충당금을 쌓는 **발생손실** 모형이었다.

!!! note "충당금이란?"
    대출을 내주면 일부는 부실이 발생한다.
    이 **예상 손실분을 미리 비용으로 잡아두는 것**이 충당금(대손충당금)이다.
    실제 부실이 발생하기 전에 재무제표에 손실을 반영하므로,
    은행의 재무 건전성을 실질적으로 보여주는 역할을 한다.

개념적으로는 \(\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}\)이지만,
실제 산출은 기간별 부도확률·손실률·익스포저를 곱해 **할인한 뒤 합산**하는 다기간 식이며,
여러 경기 시나리오의 결과를 확률 가중 평균한다.

대출은 최초 인식 이후 신용위험 변화에 따라 세 단계(Stage)로 나뉜다.

- **Stage 1** — 신용위험이 유의적으로 증가하지 않음 → 향후 **12개월** ECL
- **Stage 2** — 신용위험이 유의적으로 증가(SICR) → **전체 기간(Lifetime)** ECL
- **Stage 3** — 신용손상(부도) → 전체 기간 ECL

Stage 판정 기준(SICR), 다기간 ECL 식, 그리고 ECL에 쓰이는 PD가 **Point-in-Time(PiT)** 성격을 가져야 하는 이유는
[PD 용어 정리 — PiT PD](terms/pd.md)에서 자세히 다룬다.

<div class="source-ref">
출처: <a href="https://assets.kpmg.com/content/dam/kpmg/kr/pdf/2017/kr-ifrs-brief-201711.pdf" target="_blank">KPMG IFRS Brief (2017.11) · K-IFRS 제1109호 시행</a>
</div>

## Basel 자기자본 (IRB Approach)

내부등급법(IRB)을 사용하는 은행은 자체 추정한 위험요소로 위험가중자산(RWA)을 산출한다.

$$
\text{RWA} = f(\text{PD},\; \text{LGD},\; \text{EAD},\; M)
$$

\(f\)는 자산군(기업·리테일 등)별로 정해진 위험가중 함수이며, 자산상관계수 \(\rho\)는 은행이 추정하는 값이 아니라 함수 안에서 PD와 자산군에 따라 정해진다.
만기 \(M\)은 기업 익스포저에서 쓰이고 리테일 함수에는 들어가지 않는다.

- **기업** 익스포저는 PD만 자체 추정하고 LGD·EAD는 감독 기준값을 쓰는 기본 내부등급법(F-IRB)과, 모두 자체 추정하는 고급 내부등급법(A-IRB)으로 나뉜다.
- **리테일** 익스포저에는 F-IRB가 없다. 내부등급법을 쓰려면 PD와 LGD를 자체 추정해야 하고, 리볼빙 한도의 미사용분도 자체 EAD 추정을 써야 한다.
- 규제 PD에는 하한이 있다. 기업은 0.05%, 리테일은 적격 리볼빙(QRRE) 리볼버 0.1%, 그 밖의 리테일 0.05%이다.

RWA가 커지면 더 많은 자본을 쌓아야 하므로, PD 추정의 정확성이
은행의 **자본 효율성**에 직결된다.
등급별(소매는 풀별) PD는 기업과 소매 모두 **1년 부도율의 장기 평균**이어야 하므로
**Through-the-Cycle(TtC)** 성격이 강조되며, IFRS 9의 PiT PD와는 성격이 다르다([TtC PD](terms/pd.md)).

<div class="source-ref">
출처: <a href="https://www.bis.org/basel_framework/chapter/CRE/32.htm" target="_blank">Basel Framework CRE32.4, 32.58, 32.63 (Risk components)</a>,
<a href="https://www.bis.org/basel_framework/chapter/CRE/36.htm" target="_blank">CRE36.63 (기업), CRE36.82 (소매)</a>
</div>

## 스트레스 테스트

"경기가 심각하게 악화되면 우리 포트폴리오의 손실이 얼마나 늘어나는가?"

- 거시경제 변수(GDP, 실업률, 금리, 주택가격 등)와 PD·LGD의 관계를 위성모형(satellite model)으로 추정한다.
- 감독당국이 공통 시나리오를 제시하고 은행이 손실·자본비율 변화를 추정하는 감독 스트레스 테스트와, 은행이 자체 시나리오로 수행하는 내부 스트레스 테스트가 있다.
- 결과는 자본 계획과 위험 한도 설정에 반영된다.

이 영역에서도 PD의 **수준과 경기 민감도**가 핵심이므로, 서열만 맞는 모형으로는 답을 낼 수 없다.

## 모형 정기 검증

내부등급법 은행은 PD·LGD·EAD 추정치를 **최소 연 1회 이상** 재검토해야 한다.
신용평가모형 전반의 검증 체계(미국 SR 26-2(구 SR 11-7), 국내 감독 요건)는 [규제 프레임워크](../scorecard/part5_scorecard/regulatory-framework.md)에서 다룬다.

- **변별력 검증** — KS, AR 추이 ([성능 평가](../scorecard/part5_scorecard/performance-ks-ar-gini.md))
- **안정성 검증** — PSI, 등급 이동 행렬 ([모니터링](../scorecard/part5_scorecard/monitoring-and-operations.md))
- **정확성 검증(캘리브레이션)** — 등급별 예측 PD vs 실현 부도율(Back-testing)

<div class="source-ref">
출처: <a href="https://www.bis.org/basel_framework/chapter/CRE/36.htm" target="_blank">Basel Framework CRE36.66 ("Banks must review their estimates on a yearly basis or more frequently")</a>
</div>
