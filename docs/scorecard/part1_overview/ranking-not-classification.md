# 분류가 아닌 서열화

CSS는 이진 분류(Binary Classification) 문제를 풀지만, 그 **목적**은 일반적인 분류 모형과 근본적으로 다르다. Kaggle 같은 머신러닝 경진대회에서 이진 분류 모형을 평가할 때는 [Precision, Recall](https://en.wikipedia.org/wiki/Precision_and_recall){:target="_blank"}, [F1-Score](https://en.wikipedia.org/wiki/F-score){:target="_blank"} 등의 지표를 사용하는 것이 일반적이다. 그러나 신용평가 현업에서는 이러한 지표를 거의 사용하지 않으며, 대신 **[KS(Kolmogorov-Smirnov)](../part5_scorecard/performance-ks-ar-gini.md)**, **[AR(Accuracy Ratio, Gini)](../part5_scorecard/performance-ks-ar-gini.md)** 등의 지표로 모형을 평가한다. 이 차이는 단순한 관행이 아니라, CSS가 추구하는 목적 자체가 다르기 때문이다.

??? quote "통계량 참고 링크"
    **분류 지표** (본 가이드북에서는 다루지 않음)

    - [Precision & Recall — Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall){:target="_blank"}
    - [F-Score (F1-Score) — Wikipedia](https://en.wikipedia.org/wiki/F-score){:target="_blank"}
    - [Confusion Matrix — Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix){:target="_blank"}

    **서열화 지표** (본 가이드북에서 상세히 다룸)

    - [KS · AR · Gini — 본 가이드북 Part 5](../part5_scorecard/performance-ks-ar-gini.md)

## 2.1 분류 vs. 서열화: 목적의 차이

일반적인 이진 분류 모형은 "이 건이 Positive인가, Negative인가"를 **맞히는 것**이 목적이다. 반면 CSS의 목적은 개별 건의 정답을 맞히는 것이 아니라, 차주 간의 **리스크 순서를 올바르게 매기는 것** — 즉 **서열화(Rank-Ordering)**에 있다.

| | 일반 이진 분류 (예: Kaggle) | 신용평가모형 (CSS) |
|---|---|---|
| **목적** | 개별 건의 정답 맞히기 (0/1 분류) | 차주 간 리스크 서열화 (Rank-Ordering) |
| **핵심 질문** | "이 고객은 부실인가?" | "고객 A가 고객 B보다 더 위험한가?" |
| **모형 출력의 활용** | 예측 레이블 그 자체 | 확률 → 점수 → 등급 → 차등 관리 |
| **의사결정 구조** | 단일 Threshold로 분류 | 등급 구간별 다수 Cutoff로 차등 정책 |
| **대표 평가지표** | Precision, Recall, F1-Score | KS, AR(Gini), AUROC |

CSS가 서열화를 추구하는 이유는 모형의 **비즈니스 활용 방식**에 있다. 금융기관은 모형 점수를 기반으로 차주를 등급화하고, 등급별로 금리·한도·심사 기준·사후 관리 강도를 차등 적용한다. 이 파이프라인이 작동하려면 모형이 "누가 부실인가"를 맞히는 것보다, **"누가 더 위험한가"를 일관되게 구분하는 것**이 훨씬 중요하다.

> **불량 확률** \(\hat{p}\) → **신용 점수**(Score) → **신용 등급**(Grade) → **차등 리스크 관리**
>
> - 금리 차등 (Risk-Based Pricing)
> - 한도 차등
> - 심사 기준 차등
> - 사후 모니터링 강도 차등

## 2.2 왜 Precision · Recall · F1이 적합하지 않은가

Precision, Recall, F1-Score는 모두 **특정 Threshold를 기준으로 0/1로 분류한 결과**를 평가하는 지표다. 이 지표들이 CSS에 적합하지 않은 이유는 다음과 같다.

<div class="numbered-title">① Threshold 의존성</div>

이 지표들은 "어디서 자르느냐"에 따라 값이 완전히 달라진다. 그러나 CSS는 단일 Cutoff로 승인/거절만 하는 모형이 아니다. 1등급부터 10등급까지 **연속적인 등급 체계**를 운영하면서 등급마다 다른 정책을 적용하므로, 특정 Threshold에 종속된 지표로는 모형 자체의 변별력을 측정할 수 없다.

<div class="numbered-title">② 서열화 품질을 보장하지 않음</div>

F1-Score가 높은 모형이라 해도, 점수 순으로 정렬했을 때 등급별 부실률이 뒤집힐(비단조적일) 수 있다. 예를 들어, 3등급 부실률이 4등급보다 높은 **역전 현상**이 발생해도 F1에는 반영되지 않는다. CSS에서 이러한 역전은 등급 체계의 신뢰성을 무너뜨리는 치명적 문제다.

<div class="numbered-title">③ 실무적 의미의 부재</div>

"Recall 90%"라는 수치는 신용평가 실무에서 직접적인 의사결정 근거가 되지 못한다. 실제 운영에서는 단일 Cutoff가 아니라 등급 체계 전체가 작동하기 때문이다.

## 2.3 왜 KS · AR(Gini)를 사용하는가

CSS에서 사용하는 변별력 지표들은 공통적으로 **Threshold에 의존하지 않고**, 모형의 **전체 점수 분포**에서 Good과 Bad를 얼마나 잘 분리하는지를 측정한다.

| 지표 | 측정 대상 | 의미 |
|------|-----------|------|
| **KS** | Good·Bad 누적분포의 최대 괴리 | 두 집단이 가장 크게 벌어지는 지점의 분리도 |
| **AR (Gini)** | CAP 곡선 기반, 완전모형 대비 서열화 능력 | 점수순 정렬 시 Bad가 얼마나 앞쪽에 집중되는가 |
| **AUROC** | ROC 곡선 아래 면적 | 임의의 Good-Bad 쌍에서 Bad에 더 높은 리스크를 부여할 확률 |

### 각 지표의 핵심 직관

**KS (Kolmogorov-Smirnov)**는 전체 고객을 점수 순으로 정렬한 뒤, Good의 누적분포와 Bad의 누적분포를 그렸을 때 **두 곡선이 가장 크게 벌어지는 지점의 거리**다. 이 거리가 클수록 모형이 Good과 Bad를 잘 갈라놓고 있다는 뜻이다. 특히 **승인/거절의 cutoff를 어디에 설정할 것인가**를 결정할 때 직관적으로 연결되는 지표다.

**AR (Accuracy Ratio) / Gini**는 점수 순으로 고객을 정렬했을 때, **상위(고위험) 소수만 선발해도 실제 Bad를 얼마나 빠르게 포착하는가**를 전체 구간에 걸쳐 면적으로 측정한다. KS가 "최대 분리 **지점** 하나"에 집중하는 반면, AR은 **전체 점수 범위의 누적 서열화 품질**을 요약한다. 등급 체계 전반의 변별력을 하나의 숫자로 평가할 때 적합하다.

**AUROC**는 임의로 Good 한 명과 Bad 한 명을 뽑았을 때, 모형이 **Bad에게 더 높은 위험 점수를 부여할 확률**이다. AR과는 \(\text{AR} = 2 \times \text{AUC} - 1\)의 선형 관계를 가지므로 본질적으로 같은 정보를 담고 있다.

!!! tip "소매는 KS, 기업은 AR — 업권별 주력 지표가 다른 이유"
    **소매(Retail)** 포트폴리오는 수십만~수백만 건의 대량 자동 심사에서 **단일 cutoff**(승인/거절 기준점)가 핵심이므로, 최대 분리 지점을 직접 보여주는 **KS**를 1차 지표로 삼는다. **기업(Wholesale)** 포트폴리오는 등급별 금리·한도 차등이 핵심이고, 수백~수천 건의 소표본에서 단일 지점 의존이 불안정하므로 전체 면적을 요약하는 **AR**을 선호한다.

    다만, 실무에서 모형 성능 보고서를 작성할 때는 소매든 기업이든 **KS · AR · AUC를 모두 산출하여 보고하는 것이 관례**다. 세 지표를 함께 제시해야 모형의 변별력을 다각도로 검증할 수 있고, 감독당국·내부 검증 부서의 요구사항도 충족된다.

    각 지표의 수식, 차트, 해석 기준은 **[성능 평가 (KS·AR·Gini)](../part5_scorecard/performance-ks-ar-gini.md)**에서 상세히 다룬다.

이 지표들이 높다는 것은 **점수가 높은 집단일수록 일관되게 부실률이 낮다**는 것을 의미하며, 이는 곧 등급화의 전제 조건인 **단조성(Monotonicity)**이 유지될 가능성이 높다는 뜻이다.

$$
\text{1등급 부실률} < \text{2등급 부실률} < \cdots < \text{10등급 부실률}
$$

이 단조성이 확보되어야 등급별 차등 관리가 의미를 가지며, KS·AR은 바로 이 서열화 품질을 직접적으로 측정하는 지표다.

!!! note "분류 지표가 완전히 무의미한 것은 아니다"
    Precision, Recall 등의 지표가 CSS에서 전혀 쓰이지 않는 것은 아니다. 예를 들어, 승인/거절의 **단일 Cutoff를 설정**할 때 해당 지점에서의 True Positive Rate(= Recall)이나 오분류 비용을 참고할 수 있다. 그러나 이는 모형의 전체적인 변별력을 평가하는 용도가 아니라, 특정 운영 시점의 의사결정을 보조하는 역할에 한정된다. 모형 자체의 품질은 KS·AR 등 서열화 지표로 판단한다.
