---
title: Preface
hide:
  - toc
---

<div class="hero" markdown>

본 가이드북은 신용평가모형 개발의 두 가지 접근법을 다룹니다.
**전통적인 로지스틱 회귀 기반 스코어카드**와
**머신러닝 기반 신용평가모형** — 각각의 이론, 방법론, 실무 적용을
정리하였습니다.

전통 스코어카드 파트에서는 변수의 구간화(Classing)부터 WoE/IV 변환,
단변량·다변량 로지스틱 회귀, 스코어카드 변환과 등급화,
OOT 검증, 모니터링, 규제 프레임워크까지 전 과정을 순서대로 다루며, 머신러닝 파트에서는 트리 기반 모델,
피처 엔지니어링, 해석 가능성 등 실무 관점의 주제를 정리하였습니다.

본 가이드북은 저 혼자서 Claude Code와 공부를 하고,
내용 숙지를 위해 정리한 내용이니 틀린 내용이 있을 수 있다는 점을 감안하여
살펴봐주시기 바랍니다.

</div>

## 저자 소개

<div class="author-intro" markdown>

**[(주)씨즈데이터](https://seizedata.com/)**는 통장·카드 거래내역, 금융/공공 마이데이터 등의 원천 데이터를
신용평가에 활용 가능한 대안정보로 가공하는 데이터 전문기업입니다. 자체 개발한 DaZoom 솔루션을 통해
금융 거래 데이터를 정교하게 카테고리/항목화하고, 이를 기반으로 대안신용평가, 현금흐름 분석 등 다양한
금융 비즈니스에 적용 가능한 데이터 제품을 제공하고 있습니다.

저는 2018년부터 씨즈데이터 데이터 분석실 소속으로,
CB정보와 대안정보 기반의 신용평가모형 개발을 진행하고 있습니다.

<span class="author-badge">정승욱</span>

</div>

## 가이드북 구성

<div class="card-grid" markdown>

<div class="card-section-header">
<span class="section-badge section-badge--traditional">Traditional</span>
전통 스코어카드
</div>

<div class="card" markdown>
<span class="card-part">① 개요</span>
<div class="card-title"><a href="scorecard/part1_overview/">신용평가모형 개요</a></div>
<div class="card-desc">CSS의 정의·목적, 분류가 아닌 서열화, 모형 분류(AS/BS/Collection), Target 정의, 성과 기간 설정, 개발 프로세스 Overview</div>
</div>

<div class="card" markdown>
<span class="card-part">② 이론</span>
<div class="card-title"><a href="scorecard/part2_theory/">로지스틱 회귀의 수학적 기반</a></div>
<div class="card-desc">이진 분류 문제 정의, Odds와 Logit 변환, Sigmoid 함수, 최대우도추정(MLE)</div>
</div>

<div class="card" markdown>
<span class="card-part">③ 변수 선정</span>
<div class="card-title"><a href="scorecard/part3_variable_selection/">Classing · WoE/IV · 단변량 LR · 정보영역별 선정</a></div>
<div class="card-desc">연속형 변수 구간화(Classing), WoE 변환과 IV 평가, 단변량 로지스틱 회귀 유의성 검정, 정보영역별 Partial LR을 통한 대표 변수 확정</div>
</div>

<div class="card" markdown>
<span class="card-part">④ 모델링</span>
<div class="card-title"><a href="scorecard/part4_modeling/">로지스틱 회귀 Full Model</a></div>
<div class="card-desc">Simple LR vs Full Model, 다운샘플링, 다변량 회귀로 최종 모형 적합</div>
</div>

<div class="card" markdown>
<span class="card-part">⑤ 스코어카드</span>
<div class="card-title"><a href="scorecard/part5_scorecard/">변환 · 검증 · 모니터링 · 규제</a></div>
<div class="card-desc">스코어카드 변환·등급화, KS·AR·Gini 성능 평가, OOT 검증, PSI·CSI 모니터링과 리캘리브레이션, SR 11-7·Basel·금감원 규제 프레임워크</div>
</div>

<div class="card" markdown>
<span class="card-part">부록</span>
<div class="card-title"><a href="scorecard/appendix/">보충 자료</a></div>
<div class="card-desc">optbinning 실무 가이드, WoE 직접 투입 vs 더미 변수 비교, Stepwise 변수 선택, Score·Wald·LR 검정 이론, LOWESS 기반 미니모델링</div>
</div>

<div class="card-section-header">
<span class="section-badge section-badge--ml">ML</span>
머신러닝 기반 신용평가
</div>

<div class="card card--ml" markdown>
<span class="card-part">① ML 기초</span>
<div class="card-title"><a href="ml/part1_overview/">왜 ML인가 · Bias-Variance · 정규화 · 피처 엔지니어링</a></div>
<div class="card-desc">전통 스코어카드의 한계, EPE 분해, Ridge/Lasso 정규화, 데이터 분리 전략, 범주형 인코딩, 피처 선택</div>
</div>

<div class="card card--ml" markdown>
<span class="card-part">② 뉴럴넷</span>
<div class="card-title"><a href="ml/part2_neural_net/">신경망 기초 · TabNet · CNN/RNN</a></div>
<div class="card-desc">퍼셉트론, LR = 단일 뉴런, TabNet, 사전 가정(Inductive Bias), CNN/RNN의 정형 데이터 한계, LSTM 실무 테스트</div>
</div>

<div class="card card--ml" markdown>
<span class="card-part">③ 트리 앙상블</span>
<div class="card-title"><a href="ml/part3_tree_ensemble/">CART · RF · Boosting · XGB/LGBM</a></div>
<div class="card-desc">트리 분할, Bagging, Gradient Boosting, XGBoost/LightGBM, 하이퍼파라미터 튜닝</div>
</div>

<div class="card card--ml" markdown>
<span class="card-part">④ 해석과 설명</span>
<div class="card-title"><a href="ml/part4_interpretation/">SHAP · 1-Depth GBM · EBM · fANOVA</a></div>
<div class="card-desc">해석 도구 총람, SHAP 이론, 1-Depth GBM 스코어카드, EBM(GA²M), fANOVA와 Purification</div>
</div>

<div class="card card--ml" markdown>
<span class="card-part">⑤ 모델 검증</span>
<div class="card-title"><a href="ml/part5_validation/">성능 지표 · OOT · 규제 프레임워크</a></div>
<div class="card-desc">AUC/KS/Gini, OOT 검증, PSI/CSI 모니터링, SR 11-7, EU AI Act, 한국 금융 AI 가이드라인</div>
</div>

<div class="card card--ml" markdown>
<span class="card-part">부록</span>
<div class="card-title"><a href="ml/appendix/">보충 자료</a></div>
<div class="card-desc">SHAP과 Functional ANOVA 심화 비교, Global 해석 방법론(EBM)</div>
</div>

<div class="card-section-header">
<span class="section-badge section-badge--lending">여신</span>
여신 프로세스
</div>

<div class="card card--lending" markdown>
<span class="card-part">① 모형 개발</span>
<div class="card-title"><a href="lending_process/model-development/">가이드북의 핵심 영역</a></div>
<div class="card-desc">데이터 준비, 변수 처리, 모형 학습, 성능 평가, Score 변환 — 가이드북 각 파트와의 매핑</div>
</div>

<div class="card card--lending" markdown>
<span class="card-part">② 심사·승인</span>
<div class="card-title"><a href="lending_process/underwriting/">모형이 실무에 꽂히는 접점</a></div>
<div class="card-desc">승인/거절 의사결정, 금리 산정(Risk-Based Pricing), 한도 산정</div>
</div>

<div class="card card--lending" markdown>
<span class="card-part">③ 사후 관리</span>
<div class="card-title"><a href="lending_process/portfolio-mgmt/">실행 이후의 모니터링</a></div>
<div class="card-desc">Behavioral Scoring, 한도 증감, 조기경보(EWS), Cross-sell/Up-sell</div>
</div>

<div class="card card--lending" markdown>
<span class="card-part">④ 부실·추심</span>
<div class="card-title"><a href="lending_process/collection/">연체 발생 후의 세계</a></div>
<div class="card-desc">연체 단계 관리, Collection Scoring, 채권 매각·상각</div>
</div>

<div class="card card--lending" markdown>
<span class="card-part">⑤ 리스크·규제</span>
<div class="card-title"><a href="lending_process/risk-regulation/">PD가 '금액'이 되는 영역</a></div>
<div class="card-desc">IFRS 9 ECL, Basel 자기자본(IRB), 스트레스 테스트, 모형 정기 검증</div>
</div>

<div class="card card--lending" markdown>
<span class="card-part">용어 정리</span>
<div class="card-title"><a href="lending_process/terms/">PD · LGD · EAD</a></div>
<div class="card-desc">신용원가(EL)의 세 축 — 모형 산출물 p̂부터 TtC PD, PiT PD까지 용어 층위 정리</div>
</div>

</div>
