# 규제 프레임워크

---

## 2.1 모형 검증 시 SHAP 활용

| 검증 항목 | SHAP 활용 방법 |
|----------|---------------|
| **변수 방향성** | Summary Plot에서 "연체↑ → 부도확률↑" 등 상식과 일치하는지 확인 |
| **비합리적 패턴** | 특정 변수가 상식에 반하는 방향으로 기여하면 원인 조사 |
| **변수 기여도 집중** | 소수 변수에 SHAP이 편중되면 모형 안정성 우려 |
| **차별 변수 점검** | 성별, 연령 등 민감 변수의 SHAP 분포 확인 |

!!! warning "해석 가능성은 성능과 별개의 요구사항"
    해석이 가능하다고 좋은 모형이 아니고, 성능이 좋다고 해석이 되는 것도 아니다. 신용평가에서는 **둘 다 만족**해야 한다.

---

## 2.2 주요 규제 지침

### SR 11-7: 모형 리스크 관리의 근간 (미국)

SR 11-7의 핵심 원칙(개념적 건전성, 독립적 검증, 거버넌스 등)과 Basel IRB 검증 요건 등 **공통 규제 프레임워크**는 스코어카드 섹션의 [규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에서 상세히 다루었다. 여기서는 **ML 모형에서 추가로 고려할 사항**에 집중한다.

SR 11-7 관점에서 ML 모형은 "개념적 건전성"을 설명하기 어렵다는 도전이 있다. 1-Depth GBM / EBM은 이 도전을 완화한다:

- 각 변수의 효과를 독립적으로 시각화하고 설명할 수 있음
- 단조성 제약을 걸면 도메인 지식과 일관성 검증이 용이
- Adverse action (신용 거절 사유) 산출이 직관적

### EU AI Act: 신용평가 AI는 "고위험"

EU AI Act는 신용평가 AI를 **고위험(high-risk)** 시스템으로 분류한다. 투명성, 설명 가능성, 데이터 거버넌스, 인적 감독, 문서화가 요구된다. GAM 구조의 1-Depth GBM은 모형의 의사결정 로직이 본질적으로 투명하므로, 이 요건을 자연스럽게 충족한다.

### 한국 금융 AI 가이드라인

금융위원회(FSC)의 7대 원칙과 금감원 AI RMF의 상세 내용은 [규제 프레임워크](../../scorecard/part5_scorecard/regulatory-framework.md)에서 다루었다. ML 모형을 신용평가에 적용할 경우, 특히 **신뢰성(정확성·안정성·재현성)** 원칙이 핵심 쟁점이 된다.

### EBA ML for IRB Report (2023)

유럽은행감독청(EBA)은 IRB용 ML 사용 은행을 대상으로 실태를 조사했다. 주요 결과:

- 은행의 **40%가 Shapley values** 활용
- 20%가 시각적 도구
- 28%가 문서화 강화

### ECB Internal Models Guide (2025)

ECB는 ML 챕터를 신설하여, LIME 등 surrogate를 허용 가능한 해석 접근으로 명시했다.

---

## 2.3 해석 가능성의 비용

!!! example "정량적 비교"
    Reacfin (2023, Springer)은 해석 가능 모형(LR/GAM)과 Black-box 모형(XGBoost/NN) 간 성능 차이를 정량화했다. 50,000건 이상의 신용 익스포저에서, **해석 가능 모형만 사용할 때의 비용은 연간 ROI 기준 약 15~20bp**였다. 이 차이가 surrogate 접근의 운영 복잡성을 정당화하는지는 기관마다 판단이 다르다.

---

## 2.4 규제의 방향

!!! note "핵심 메시지"
    한국을 포함한 주요 규제 당국의 방향은 명확하다: **설명 가능하지 않은 AI 모형은 점점 더 운영하기 어려워진다.** 1-Depth GBM / EBM 같은 inherently interpretable 모형은 별도의 사후 해석(post-hoc explanation) 없이도 규제 요건을 충족할 수 있어, 모형 리스크 관리 비용을 줄여준다.

한편 [왜 ML인가](../part1_overview/why_ml.md)에서 다룬 바와 같이, 국내에서는 **규제모형은 전통 스코어카드, ML은 Cross Matrix 오버라이드 전략모형**이라는 이원 체계가 현실적이다. ML이 규제모형 자체를 대체하는 것이 아니라, 기존 프레임 위에 ML의 변별력을 얹는 하이브리드 접근이 주류다.

---

## 2.5 참고 자료

| 자료 | 유형 | 내용 |
|------|------|------|
| **[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)** (Christoph Molnar) | 무료 온라인 서적 | PDP, SHAP, LIME 등 전 범위. **가장 추천** |
| **[SHAP GitHub](https://github.com/shap/shap)** | 공식 저장소 | TreeSHAP API, 시각화 예제, 튜토리얼 노트북 |
| **[Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)** | 논문 (NeurIPS) | "A Unified Approach to Interpreting Model Predictions" — SHAP 원논문 |
| **[Lundberg et al. (2020)](https://www.nature.com/articles/s42256-019-0138-9)** | 논문 (Nature MI) | "From Local Explanations to Global Understanding with Explainable AI for Trees" — TreeSHAP |
| **[Dumitrescu et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0377221721005695)** | 논문 (EJOR) | "Machine Learning for Credit Scoring" — PLTR |
| **[Sudjianto & Zhang (2021)](https://www.semanticscholar.org/paper/Designing-Inherently-Interpretable-Machine-Learning-Sudjianto-Zhang/90409ae91767248e9ea88b7d6ab44e18f0e1a9be)** | 논문 | "Designing Inherently Interpretable ML Models" — Wells Fargo |
| **[EBA ML for IRB Report (2023)](https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-follow-report-use-machine-learning-internal)** | 규제 보고서 | 은행의 ML 활용 현황 및 해석 가능성 접근 조사 |

!!! tip "이전 섹션"
    [해석과 설명](../part4_interpretation/index.md)에서 SHAP, 1-Depth GBM, EBM 등 ML 해석 기법을 다루었다.
