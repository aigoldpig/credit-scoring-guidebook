# 규제 프레임워크

신용평가모형 전반에 적용되는 공통 규제(미국 모형리스크 관리 지침 SR 26-2(구 SR 11-7), Basel IRB 검증 요건, 국내 모형 검증 요건, 금융분야 AI 가이드라인, EBA·ECB 동향)는 스코어카드 섹션의 [규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에서 정리했다. 이 페이지는 **ML 모형을 신용평가에 쓸 때 특히 문제가 되는 규제**와 그 대응 방법에 집중한다.

---

## 2.1 국내: 자동화평가에 대한 설명요구권

### 신용정보법 제36조의2

「신용정보의 이용 및 보호에 관한 법률」 제36조의2(자동화평가 결과에 대한 설명 및 이의제기 등)는 2020년 2월 개정으로 신설되었다. 핵심 내용은 다음과 같다.

- **설명 요구(제1항)**: 개인인 신용정보주체는 개인신용평가회사와 대통령령으로 정하는 신용정보제공·이용자에게 다음을 설명하도록 요구할 수 있다.
    1. 개인신용평가, 금융거래의 설정·유지 및 내용 결정 등에 **자동화평가를 하는지 여부**
    2. 자동화평가를 하는 경우 그 **결과**, **주요 기준**, **기초정보의 개요** 등
- **정보 제출·정정·재산출 요구(제2항)**: 자동화평가 결과에 유리하다고 판단되는 정보를 제출하거나, 기초정보가 부정확하거나 최신이 아니라고 판단되면 정정·삭제를 요구하거나 **평가 결과를 다시 산출하도록** 요구할 수 있다.
- 법령이 정한 사유가 있으면 요구를 거절할 수 있으며, 요구 절차·방법과 거절 통지 등 세부 사항은 대통령령(시행령)에 위임되어 있다.

<div class="source-ref">출처: <a href="https://www.law.go.kr/법령/신용정보의이용및보호에관한법률" target="_blank">국가법령정보센터 「신용정보의 이용 및 보호에 관한 법률」</a> 제36조의2 [본조신설 2020. 2. 4.]</div>

### ML 모형에서 이 조항이 의미하는 것

| 요구 | 스코어카드 | ML 모형에서의 과제 |
|---|---|---|
| **주요 기준 설명** | 점수표에서 감점이 큰 구간이 주요 기준(감점의 기준점과 동점 처리 규칙은 미리 정한다) | 개인별 SHAP 값 등으로 예측을 변수별 기여로 분해해, 위험을 높인 상위 변수를 사유로 제시해야 한다. 상관 변수끼리 기여가 나뉘거나 엇갈린 부호로 나타나면 사유가 어색해지므로, 사유 코드는 변수 그룹 단위로 묶어 설계하는 것이 안전하다 |
| **사유의 일관성** | 같은 입력이면 같은 사유 | 사유 산출 방식(SHAP의 기준값·배경 데이터, 계산 모드)을 고정·문서화해야 같은 고객에게 같은 사유가 나온다 |
| **재산출** | 정정된 정보로 점수표 재계산 | 운영 모형과 전처리를 그대로 재현할 수 있어야 한다. 모형 파일·라이브러리 버전·전처리 고정(재현성)이 법적 요구와 직결된다 |
| **기초정보 개요** | 사용 변수 목록이 짧고 명확 | 변수 수가 많을수록 "기초정보의 개요"를 소비자가 이해할 수 있는 수준으로 요약하는 부담이 커진다 |

### 인공지능 기본법

「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」(인공지능 기본법)이 2026년 1월 22일 시행되었다. 이 법은 사람의 생명·안전·기본권에 중대한 영향을 미치거나 위험을 초래할 우려가 있는 영역에서 쓰이는 AI를 **고영향 인공지능**으로 정의한다. 대상 영역에는 **대출 심사 등 개인의 권리·의무 관계에 중대한 영향을 미치는 판단 또는 평가**가 포함된다.

고영향 인공지능에 해당하면 사업자는 위험관리방안, 결과 도출에 쓰인 주요 기준 등에 대한 설명방안, 이용자 보호방안, 사람의 관리·감독, 안전성·신뢰성 확보 조치를 확인할 수 있는 문서의 작성·보관 등의 책무를 진다. 해당 여부는 사업자가 사전에 검토해야 하며, 판단이 어려우면 과학기술정보통신부에 확인을 요청할 수 있다. 시행 초기에는 정부가 과태료 부과를 유예하는 계도기간을 운영하고 있으므로, 세부 판단 기준과 의무 범위는 시행령과 정부 가이드라인의 최신본으로 확인해야 한다.

<div class="source-ref">출처: <a href="https://www.law.go.kr/법령/인공지능발전과신뢰기반조성등에관한기본법" target="_blank">국가법령정보센터 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」</a> (2026. 1. 22. 시행)</div>

### 금융분야 AI 가이드라인과 감독 규정

금융위원회의 금융분야 AI 가이드라인(7대 원칙)과 금융감독원의 AI 위험관리 프레임워크, 모형 검증 요건은 [스코어카드 규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에 정리했다. ML 모형에서는 그중 **신뢰성(정확성·안정성·재현성)** 원칙이 가장 직접적인 쟁점이 된다. [모델 검증과 모니터링](validation.md)의 시드 민감도, SHAP 안정성, 재현성 점검이 이 원칙에 대한 실무적 대응이다.

---

## 2.2 해외

### SR 26-2 (미국, 구 SR 11-7)

미국의 모형리스크 관리 지침 SR 11-7(2011)은 2026년 4월 17일 SR 26-2로 대체되었다. 두 지침의 차이와 핵심 원칙(개념적 건전성, 독립적 검증, 거버넌스 등)은 [스코어카드 규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에서 다룬다. ML 모형은 수백 개 트리의 합이라 "개념적 건전성", 즉 모형이 왜 그렇게 작동하는지를 설명하기 어렵다는 것이 가장 큰 과제다. 변수별 방향 점검, 단조 제약, 챌린저 비교, 민감도 분석 같은 [ML 특화 검증](validation.md)이 이 설명 부담을 나누어 진다. 구조 자체를 해석 가능하게 만든 모형([1-Depth GBM](../part4_interpretation/depth1_gbm.md), [EBM](../part4_interpretation/ebm.md))은 이 부담을 줄이는 선택지 중 하나다.

### EU AI Act

EU AI Act는 자연인의 신용도를 평가하거나 신용점수를 산정하는 AI 시스템을 **고위험(high-risk)**으로 분류한다(부속서 III, 금융 사기 탐지 목적은 제외). 고위험 시스템에는 위험관리 체계, 데이터 거버넌스, 기술 문서화, 기록 보관(로깅), 이용자에 대한 투명성, 인적 감독, 정확성·견고성 등의 요건이 적용된다. 모형이 해석 가능하다는 것은 이 중 투명성과 인적 감독 요건에 대응하기 쉽게 해 줄 뿐, 나머지 요건을 대신하지는 않는다.

적용 시점도 확인이 필요하다. AI Act는 2024년 8월 발효되었고, 부속서 III 고위험 시스템 의무는 원래 2026년 8월부터 적용될 예정이었다. 이후 EU의 AI 규제 간소화 개정(Digital Omnibus)으로 2027년 12월로 연기된 것으로 알려져 있으므로, 유럽 사업에 적용할 때는 공식 관보의 최신 일정을 확인해야 한다.

### EBA·ECB

유럽은행감독청(EBA)의 IRB 모형 ML 활용 후속 보고서(2023)와 유럽중앙은행(ECB)의 내부모형 가이드 개정(2025)은 [스코어카드 규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에서 다룬다.

---

## 2.3 해석 가능성의 비용

해석 가능한 모형만 쓰면 성능을 얼마나 포기하는가? Dessain, Bentaleb & Vinas(2023)는 신용평가 사례에서 이 질문을 경제적 수익 기준으로 정량화했다. 내재적으로 해석 가능한 모형(GLM, GAM 계열)과 블랙박스 모형을 비교한 결과, **최고 성능 블랙박스 모형과 최고 성능 해석 가능 모형의 차이는 연간 투자수익률(ROI) 기준 15~20bp**였고, 이 "해석 가능성의 비용"은 위험 선호도가 높을수록 커졌다.

이 차이가 블랙박스 모형과 사후 해석 체계를 운영하는 복잡성을 정당화하는지는 기관마다 판단이 다르다. 국내에서는 설명요구권과 고영향 AI 책무까지 고려해야 하므로, 성능 차이와 **설명·검증·운영 비용**을 함께 비교하는 것이 현실적이다.

<div class="source-ref">출처: Dessain, J., Bentaleb, N. & Vinas, F., "Cost of Explainability in AI: An Example with Credit Scoring Models", <em>Explainable Artificial Intelligence (xAI 2023)</em>, Communications in Computer and Information Science, Springer, 2023.</div>

---

## 2.4 규제의 방향과 국내 현실

!!! note "핵심 메시지"
    국내외 규제의 방향은 명확하다. **설명할 수 없는 AI 모형은 점점 더 운영하기 어려워진다.** 설명 가능성은 성능과 별개의 요구사항이다. 해석이 가능하다고 좋은 모형이 아니고, 성능이 좋다고 설명이 되는 것도 아니다. 신용평가에서는 **둘 다** 만족해야 한다.

[왜 ML인가](../part1_overview/why_ml.md)에서 다룬 것처럼, 국내에서는 **규제모형은 전통 스코어카드, ML은 Cross Matrix 오버라이드 등 전략모형**으로 쓰는 이원 체계가 현실적이다. ML이 규제모형 자체를 대체하기보다, 기존 프레임 위에 ML의 변별력을 얹는 하이브리드 접근이 주류다. ML을 심사 의사결정에 직접 쓰는 순간 설명요구권과 고영향 AI 책무의 대상이 되므로, 전략모형이라도 이 페이지의 요건을 염두에 두고 설계해야 한다.

---

## 2.5 참고 자료

| 자료 | 유형 | 내용 |
|------|------|------|
| **[신용정보의 이용 및 보호에 관한 법률](https://www.law.go.kr/법령/신용정보의이용및보호에관한법률)** | 법률 | 제36조의2 자동화평가 결과에 대한 설명 및 이의제기 |
| **[인공지능 발전과 신뢰 기반 조성 등에 관한 기본법](https://www.law.go.kr/법령/인공지능발전과신뢰기반조성등에관한기본법)** | 법률 | 고영향 인공지능의 정의와 사업자 책무 |
| **[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)** (Christoph Molnar) | 무료 온라인 서적 | PDP, SHAP, LIME 등 전 범위 |
| **[SHAP GitHub](https://github.com/shap/shap)** | 공식 저장소 | TreeSHAP, 시각화 예제, 튜토리얼 |
| **[Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)** | 논문 (NeurIPS) | "A Unified Approach to Interpreting Model Predictions" — SHAP 원논문 |
| **[Lundberg et al. (2020)](https://www.nature.com/articles/s42256-019-0138-9)** | 논문 (Nature Machine Intelligence) | "From Local Explanations to Global Understanding with Explainable AI for Trees" — TreeSHAP |
| **[Dumitrescu et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377221721005695)** | 논문 (EJOR) | "Machine Learning for Credit Scoring: Improving Logistic Regression with Non-linear Decision-tree Effects" — PLTR |
| **[Sudjianto & Zhang (2021)](https://www.semanticscholar.org/paper/Designing-Inherently-Interpretable-Machine-Learning-Sudjianto-Zhang/90409ae91767248e9ea88b7d6ab44e18f0e1a9be)** | 논문 | "Designing Inherently Interpretable Machine Learning Models" |
| **[Dessain et al. (2023)](https://link.springer.com/chapter/10.1007/978-3-031-44064-9_26)** | 논문 (xAI 2023, CCIS) | "Cost of Explainability in AI: An Example with Credit Scoring Models" |
| **[EBA ML for IRB Report (2023)](https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-follow-report-use-machine-learning-internal)** | 규제 보고서 | IRB 모형의 ML 활용 현황 및 해석 가능성 접근 |

!!! tip "이전 섹션"
    [해석과 설명](../part4_interpretation/index.md)에서 SHAP, 1-Depth GBM, EBM 등 ML 해석 기법을 다루었다.
