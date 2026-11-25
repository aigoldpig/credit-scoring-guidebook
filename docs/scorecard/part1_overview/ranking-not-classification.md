# 분류가 아닌 서열화

CSS는 이진 분류(Binary Classification) 문제를 풀지만, 그 **목적**은 일반적인 분류 모형과 근본적으로 다르다. Kaggle 같은 머신러닝 경진대회에서 이진 분류 모형을 평가할 때는 [Precision, Recall](https://en.wikipedia.org/wiki/Precision_and_recall){:target="_blank"}, [F1-Score](https://en.wikipedia.org/wiki/F-score){:target="_blank"} 등의 지표를 사용하는 것이 일반적이다. 그러나 신용평가 현업에서는 이러한 지표를 모형 평가에 거의 사용하지 않으며, 대신 **[KS(Kolmogorov-Smirnov)](../part5_scorecard/performance-ks-ar-gini.md)**, **[AR(Accuracy Ratio, = Gini)](../part5_scorecard/performance-ks-ar-gini.md)** 등의 지표로 모형을 평가한다. 이 차이는 단순한 관행이 아니라, CSS가 추구하는 목적 자체가 다르기 때문이다.

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
| **대표 평가지표** | Precision, Recall, F1-Score | KS, AR(= Gini) |

CSS가 서열화를 추구하는 이유는 모형의 **비즈니스 활용 방식**에 있다. 금융기관은 모형 점수를 기반으로 차주를 등급화하고, 등급별로 금리·한도·심사 기준·사후 관리 강도를 차등 적용한다. 이 파이프라인이 작동하려면 모형이 "누가 부실인가"를 맞히는 것보다, **"누가 더 위험한가"를 일관되게 구분하는 것**이 훨씬 중요하다.

> **불량 확률** \(\hat{p}\) → **신용 점수**(Score) → **신용 등급**(Grade) → **차등 리스크 관리**
>
> - 금리 차등 (Risk-Based Pricing)
> - 한도 차등
> - 심사 기준 차등
> - 사후 모니터링 강도 차등

## 2.2 왜 Precision · Recall · F1이 적합하지 않은가

Precision, Recall, F1-Score는 모두 **특정 Threshold를 기준으로 0/1로 분류한 결과**를 평가하는 지표다. 이 지표들이 CSS 모형 평가에 적합하지 않은 이유는 다음과 같다.

<div class="numbered-title">① Threshold 의존성</div>

이 지표들은 "어디서 자르느냐"에 따라 값이 완전히 달라진다. 그러나 CSS는 단일 Cutoff로 승인/거절만 하는 모형이 아니다. 1등급부터 10등급까지 **연속적인 등급 체계**를 운영하면서 등급마다 다른 정책을 적용하므로, 특정 Threshold에 종속된 지표로는 모형 자체의 변별력을 측정할 수 없다.

<div class="numbered-title">② 서열화 품질을 보장하지 않음</div>

F1-Score가 높은 모형이라 해도, 점수 순으로 정렬했을 때 등급별 부실률이 뒤집힐(비단조적일) 수 있다. 예를 들어, 3등급 부실률이 4등급보다 높은 **역전 현상**이 발생해도 F1에는 반영되지 않는다. CSS에서 이러한 역전은 등급 체계의 신뢰성을 무너뜨리는 치명적 문제다.

<div class="numbered-title">③ 실무적 의미의 부재</div>

"Recall 90%"라는 수치는 신용평가 실무에서 직접적인 의사결정 근거가 되지 못한다. 실제 운영에서는 단일 Cutoff가 아니라 등급 체계 전체가 작동하기 때문이다.

## 2.3 왜 KS · AR(Gini)를 사용하는가

CSS에서 사용하는 변별력 지표들은 공통적으로 **Threshold에 의존하지 않고**, 모형의 **전체 점수 분포**에서 Good과 Bad를 얼마나 잘 분리하는지를 측정한다.

| 지표 | 측정 대상 | 의미 | 단위 |
|------|-----------|------|------|
| **KS** | Good·Bad 누적분포의 최대 괴리 | 두 집단이 가장 크게 벌어지는 지점의 분리도 | %p (0~100) |
| **AR (= Gini)** | CAP 곡선 기반, 완전모형 대비 서열화 능력 | 고위험(저점수)부터 선발할 때 Bad가 얼마나 빨리 포착되는가 | 소수 (0~1) |
| (참고) **AUC** | ROC 곡선 아래 면적 | 임의의 Good-Bad 쌍에서 Good의 점수가 Bad보다 높을 확률. AR과 1:1로 대응 | 소수 (0.5~1) |

### 각 지표의 핵심 직관

**KS (Kolmogorov-Smirnov)**는 전체 고객을 점수 순으로 정렬한 뒤, Good의 누적분포와 Bad의 누적분포를 그렸을 때 **두 곡선이 가장 크게 벌어지는 지점의 거리**다. 이 거리가 클수록 모형이 Good과 Bad를 잘 갈라놓고 있다는 뜻이다. 특히 **승인/거절의 cutoff를 어디에 설정할 것인가**를 결정할 때 직관적으로 연결되는 지표다.

**AR (Accuracy Ratio) / Gini**는 점수가 낮은(고위험) 고객부터 차례로 선발했을 때, **실제 Bad를 얼마나 빠르게 포착하는가**를 전체 구간에 걸쳐 면적으로 측정한다. KS가 "최대 분리 **지점** 하나"에 집중하는 반면, AR은 **전체 점수 범위의 누적 서열화 품질**을 요약한다. 등급 체계 전반의 변별력을 하나의 숫자로 평가할 때 적합하다.

**AUC**는 임의로 Good 한 명과 Bad 한 명을 뽑았을 때 **Good의 점수가 Bad보다 높을 확률**(= 모형이 Bad에게 더 높은 위험을 부여할 확률)이다. AR과는 \(\text{AR} = 2 \times \text{AUC} - 1\)의 관계가 있어 같은 정보를 담고 있으므로, 본 가이드북은 AR로 표기하고 AUC는 필요할 때 괄호로 병기한다.

!!! tip "업권별로 주로 보는 지표"
    실무에서는 단일 cutoff가 중요한 소매(Retail) 포트폴리오는 KS를, 등급 체계 전반의 서열이 중요하고 표본이 작은 기업(Wholesale) 포트폴리오는 AR을 1차 지표로 보는 관행이 있다. 다만 성능 보고에는 KS와 AR을 함께 제시하는 것이 일반적이다.
    그 이유와 지표별 수식·차트·해석 기준은 **[성능 평가 (KS·AR·Gini)](../part5_scorecard/performance-ks-ar-gini.md)**에서 상세히 다룬다.

KS·AR이 높다는 것은 Good과 Bad가 점수상에서 잘 분리되어 있다는 뜻이다. 이는 등급화의 전제 조건인 **단조성(Monotonicity)**이 유지될 가능성을 높여 준다.

$$
\text{1등급 부실률} < \text{2등급 부실률} < \cdots < \text{10등급 부실률}
$$

다만 KS·AR은 분포 전체를 요약한 숫자이므로, 값이 높아도 **특정 인접 등급 사이에서 부실률이 뒤집히는 국지적 역전**은 드러나지 않을 수 있다. 그래서 실무에서는 KS·AR과 함께 **등급별 부실률 표**를 직접 확인해 단조성을 따로 점검한다([OOT 검증 — 등급별 부도율 단조성](../part5_scorecard/oot-validation.md)).

!!! note "분류 지표가 완전히 무의미한 것은 아니다"
    Precision, Recall 등의 지표가 CSS 실무에서 전혀 쓰이지 않는 것은 아니다. 예를 들어, 승인/거절의 **단일 Cutoff를 설정**하거나 조기경보처럼 "몇 명에게 경보를 낼지" 정해야 할 때는 해당 지점에서의 포착률(= Recall)이나 오분류 비용을 참고할 수 있다. 그러나 이는 모형의 전체적인 변별력을 평가하는 용도가 아니라, 특정 운영 시점의 의사결정을 보조하는 역할에 한정된다. 모형 자체의 품질은 KS·AR 등 서열화 지표로 판단한다.
