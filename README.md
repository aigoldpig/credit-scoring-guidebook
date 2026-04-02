# 신용평가모형(Credit Scoring System) 가이드북

신용평가모형 개발의 두 가지 접근법 — **전통 로지스틱 회귀 기반 스코어카드**와 **머신러닝 기반 신용평가모형** — 의 이론, 방법론, 실무 적용을 정리한 가이드북입니다.

> 본 가이드북은 Claude Code와 함께 공부하며 정리한 내용입니다.
> 틀린 내용이 있을 수 있으니 참고 수준으로 봐주시기 바랍니다.

📖 **[가이드북 보러가기](https://aigoldpig.github.io/credit-scoring-guidebook/)**

---

## 가이드북 구성

### 📊 Traditional Scorecard

로지스틱 회귀 기반 전통 스코어카드 개발의 전 과정을 다룹니다.

| Part | 주제 | 핵심 내용 |
|:----:|------|-----------|
| **1** | 개요 | CSS 정의·목적, 서열화 vs 분류, 모형 분류(AS/BS/Collection), Target·성과 기간 정의, 개발 파이프라인 |
| **2** | 이론 | 이진 분류 정의, Odds·Logit·Sigmoid, 최대우도추정(MLE) |
| **3** | 변수 선정 | Classing(Fine/Coarse) → WoE 변환·IV 평가 → 단변량 LR(β ≈ −WoE 증명, 유의성 검정) → 정보영역별 선정 |
| **4** | 모델링 | Simple LR vs Full Model, 다운샘플링, 다변량 회귀 적합 |
| **5** | 스코어카드 | PDO 변환·등급화, KS·AR·Gini 성능 평가, OOT 검증, 모니터링·운영, 규제 프레임워크 |
| **부록** | 보충 | A. optbinning 실무 가이드 · B. 로지스틱 적합 비교 · C. LOWESS 미니모델링 · D. 기호 용어집 |

### 🤖 머신러닝

전통 스코어카드와 대비되는 머신러닝 접근법을 다룹니다.

| Part | 주제 | 핵심 내용 |
|:----:|------|-----------|
| **1** | ML 기초 | 왜 ML인가, Bias-Variance Tradeoff, 정규화(Ridge/Lasso), Baseline 워크플로우, 데이터 분리, 피처 전처리·선택 |
| **2** | 뉴럴넷 | 신경망 기초, LR = 단일 뉴런, TabNet, CNN·RNN |
| **3** | 트리 앙상블 | CART → Bagging/RF → Bias-Variance → Boosting 기초·심화 → XGBoost/LightGBM → 하이퍼파라미터 튜닝 |
| **4** | 해석과 설명 | 해석 가능성, SHAP 실무, Surrogate Model, 1-Depth GBM·EBM, 도구·구현 |
| **5** | 모델 검증 | 모델 검증·모니터링, 규제 프레임워크 |

---

## 기술 스택

| 도구 | 용도 |
|------|------|
| [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | 정적 사이트 생성 |
| MathJax 3 | 수식 렌더링 |
| matplotlib | Chart.js → 정적 차트 변환 |
| GitHub Pages | 배포 |
| Claude Code | 문서 작성 보조 |

## 저자

**정승욱** — (주)씨즈데이터 데이터 분석실

2018년부터 CB정보와 대안정보 기반 신용평가모형을 개발하고 있습니다.
씨즈데이터는 통장·카드 거래내역, 금융/공공 마이데이터 등 원천 데이터를 신용평가용 대안정보로 가공하는 데이터 전문기업입니다.

본 자료는 Claude Code로 작성되었으며, NICE·KCB 등 CB사 공개 자료를 참고하였습니다.
