# Stepwise 변수 선택: 단계별 동작 원리

Stepwise 변수 선택은 매 스텝마다 Forward(진입) → Backward(제거) 순서로 반복한다. 두 단계에서 서로 다른 검정을 쓰는 이유는 **추가 적합 없이 계산할 수 있는 검정을 고르기 위해서**다. 진입 후보는 아직 모형 밖에 있으므로 "제외 모형"만으로 계산되는 Score Test를, 제거 후보는 이미 모형 안에 있으므로 "포함 모형"만으로 계산되는 Wald Test를 쓴다. 검정의 수식과 원리는 [통계 검정 이론](score-wald-lr-test.md)에서 다룬다.

---

## 3.1 WoE 방식에서의 Stepwise

변수당 계수가 1개(\(\beta_j\))이므로 진입/탈락 판단이 단순하다.

**→** **Forward 진입 — Score Test,** \(\chi^2(1)\)

아직 모형에 없는 후보 변수 \(X_j\)에 대해 "현재 모형의 잔차와 \(X_j\)의 상관"을 측정한다. 재적합 없이 계산된다. 후보 중 Score 통계량이 가장 크고(=p-value가 가장 작고) p-value가 진입 기준 SLENTRY 미만인 변수 1개를 추가한다.

**←** **Backward 제거 — Wald Test,** \(\chi^2(1)\)

현재 모형에 포함된 각 변수의 \(\hat{\beta}_j\)를 표준오차로 나눠 유의성을 평가한다. Wald p-value가 가장 크고 잔류 기준 SLSTAY를 넘는 변수 1개를 제거한다.

SAS PROC LOGISTIC의 기본값은 SLENTRY = SLSTAY = 0.05다. 실무에서는 SLSTAY를 0.10처럼 SLENTRY보다 느슨하게 두기도 한다(이유는 3.6절).

!!! success "구현 단순성"
    변수당 p-value가 1개뿐이므로 구현이 간단하다. 모든 변수를 개별 컬럼으로 처리하면 된다.

---

## 3.2 더미 방식에서의 Stepwise

변수 X1이 구간 3개(더미 2개: \(\gamma_1\), \(\gamma_2\))라면, 진입/탈락은 **변수(그룹) 단위**로 이뤄진다. 개별 더미 단위가 아니다.

**→** **Forward 진입 — 결합 Score Test,** \(\chi^2\)**(구간수−1)**

"\(\gamma_1 = \gamma_2 = 0\)"이라는 결합 제약 하에서 X1 그룹 전체의 Score 통계량을 계산한다. \(\gamma_1\)이 아무리 유의해 보여도 X1 그룹의 결합 p-value가 SLENTRY를 넘으면 X1은 진입하지 못한다.

**←** **Backward 제거 — 결합 Wald Test,** \(\chi^2\)**(구간수−1)**

\(\hat{\gamma}_1\), \(\hat{\gamma}_2\)의 공분산을 반영한 행렬 연산으로 X1 그룹 전체의 Wald 통계량을 계산한다. \(\gamma_2\)가 개별적으로 비유의해도 결합 Wald p-value가 SLSTAY 이하이면 X1 전체가 잔류한다.

---

## 3.3 구간 일부만 비유의일 때의 처리

더미 방식의 실무적 핵심 이슈다. Stepwise가 X1을 잔류시켰더라도 개별 더미 계수를 확인해야 한다. 아래에서 구간 0은 기준 구간(최위험), \(\gamma_b\)는 **기준 구간 대비** 차이다.

!!! note "Case 1 — 결합 검정 유의, 그러나 \(\gamma_2\) 개별 p > 0.1"
    결합 검정이 유의하므로 X1은 잔류한다. 그러나 \(\gamma_2\)가 비유의하다는 것은 "**구간 2가 기준 구간(0)과 구분되지 않는다**"는 뜻이다. 구간 1과 구간 2의 차이를 보려면 \(H_0: \gamma_1 = \gamma_2\)를 따로 검정해야 한다(또는 기준 구간을 바꿔 재적합). Stepwise 후 이런 검토를 거쳐 **구분되지 않는 이웃 구간을 합치고(Re-classing) 재적합**한다. 기준 구간이 최위험이고 구간 번호가 위험도 순이라면, 구간 2(가장 안전)가 기준과 구분되지 않는 것은 구간 순서 자체가 흔들린다는 신호이므로 구간화를 근본적으로 다시 본다. 판단 기준은 본문 [유의성 검정](../../part3_variable_selection/univariate_lr/significance-test.md)의 정본("**이웃 Bin 간 차이**가 비유의면 합병 검토")을 따른다.

!!! success "Case 2 — 결합 검정도 비유의"
    X1 전체가 제거된다. 판단이 명확하다. 변수 선별 단계(IV, 단변량 로지스틱 회귀)를 다시 볼 필요가 있다.

!!! warning "Case 3 — 결합 검정 비유의, 그러나 \(\gamma_1\) 개별 p < 0.05"
    X1이 제거된다. 효과가 없는 \(\gamma_2\)까지 함께 검정하면서 자유도가 늘어 결합 검정이 희석된 것이다. **구간화 재설계가 필요하다.** \(\gamma_2\) 구간을 기준 구간과 합쳐 더미 수를 줄이면 결합 검정이 살아날 수 있다.

!!! danger "결론"
    **더미 방식에서 Stepwise는 변수 선택의 끝이 아니라 구간 재검토의 출발점이다.** 최종 모형 확정 전 각 변수의 더미 계수 개별 유의성, **이웃 구간 간 차이**, 부호 방향(최위험 기준이면 안전 구간일수록 \(\gamma\)가 더 작은 음수)을 점검한다.

---

## 3.4 AIC 기반 Stepwise

p-value 방식이 "이 변수의 계수가 통계적으로 0과 다른가?"를 묻는다면, AIC 방식은 **"이 변수가 있는 모형이 없는 모형보다 전체적으로 더 좋은가?"** 를 묻는다. 가설검정 없이 두 숫자의 크기만 비교한다.

$$
\text{AIC} = -2\log L + 2k \tag{1}
$$

\(-2\log L\)은 적합 부족도(작을수록 좋음), \(2k\)는 모형 복잡도 페널티다. **\(k\)는 절편을 포함한 추정 모수의 수**다. WoE 방식에서는 변수 1개가 모수 1개지만, 더미 방식에서는 변수 1개가 (구간수−1)개 모수를 쓴다. 변수를 추가하면 \(\log L\)은 감소하지 않지만(같거나 커짐) 페널티도 늘어나므로, **적합 개선이 페널티를 넘는가**로 진입/탈락을 정한다. 변수 1개(모수 1개)를 추가할 때 AIC가 줄어들 조건은 \(-2\log L\)이 2보다 많이 줄어드는 것이다.

### Forward 예시 — 변수 추가

현재 모형에 변수 3개(DSR, 카드이용률, 업력)와 절편이 있고, 후보로 부채비율을 고려하는 상황:

!!! note ""
    현재 모형 (\(k\)=4): AIC = −2 × (−320.5) + 2 × 4 = **649.0**

    후보 모형 (\(k\)=5): AIC = −2 × (−318.1) + 2 × 5 = **646.2**

    646.2 < 649.0 → 부채비율 **추가**

### Backward 예시 — 변수 제거

몇 스텝 뒤 신용조회건수까지 들어와 변수 5개(DSR, 카드이용률, 업력, 부채비율, 신용조회건수)인 모형(\(k\)=6, \(-2\log L = 630.6\))에서 각 변수를 하나씩 빼 본다.

!!! note ""
    현재 모형 (\(k\)=6): AIC = 630.6 + 12 = **642.6**

    - DSR 제거 (\(k\)=5): AIC = 651.3 → 나빠짐
    - 카드이용률 제거 (\(k\)=5): AIC = 648.7 → 나빠짐
    - 업력 제거 (\(k\)=5): AIC = 646.1 → 나빠짐
    - 부채비율 제거 (\(k\)=5): AIC = 641.8 → **좋아짐**
    - 신용조회건수 제거 (\(k\)=5): AIC = 644.9 → 나빠짐

    641.8 < 642.6 → 부채비율 **제거**

부채비율은 앞 스텝에서 AIC를 낮춰 들어왔지만, 신용조회건수가 들어온 뒤에는 두 변수가 겹치는 정보 때문에 부채비율의 추가 기여가 페널티(2)보다 작아졌다(\(-2\log L\) 증가 1.2). Forward에서 들어온 변수라도 다른 변수와 함께 있을 때 중복 정보가 많으면 Backward에서 걸러진다.

<div class="source-ref">예시 수치는 가상 값이다.</div>

### p-value vs AIC 비교

| 비교 항목 | p-value 기반 | AIC 기반 |
|-----------|------------------------|------------------------|
| **판단 기준** | 유의수준 임계값 (SLENTRY, SLSTAY) | AIC 감소 여부 |
| **가설검정** | 있음 (Score / Wald Test) | 없음 (숫자 비교만) |
| **임계값 설정** | 사람이 직접 설정 | 불필요 (페널티 계수 2가 사실상 임계값) |
| **과적합 방지** | 유의수준으로 간접 통제 | 페널티 항(\(2k\))에 내포 |
| **계수 유의성 설명** | 직관적 (p-value로 설명) | 덜 직관적 |
| **감독기관 보고** | 금융권에서 익숙한 형태 | 상대적으로 덜 사용 |
| **주요 사용처** | 국내 금융권 실무 (SAS 등) | 통계·학술 분야 (R 등) |

!!! note "참고 — 모수 1개 추가 시 AIC 기준이 뜻하는 유의수준"
    모수 1개를 더할 때 AIC가 줄어들 조건은 LR 통계량 > 2이고, \(\chi^2(1)\)에서 이는 p < 약 0.157에 해당한다. 즉 AIC 기준은 SLENTRY ≈ 0.157인 p-value 기준과 비슷하게 동작하며, p = 0.05 기준보다 변수를 더 많이 받아들인다.

!!! warning "BIC (Bayesian Information Criterion)"
    AIC의 변형으로 페널티를 더 강하게 준다.

    $$\text{BIC} = -2\log L + k \cdot \ln(n)$$

    표본 수 \(n\)이 8 이상이면 \(\ln n > 2\)이므로 AIC보다 페널티가 크고, 표본이 클수록 변수를 더 보수적으로 고른다. 신용평가처럼 표본이 수만~수십만 건이면 BIC는 매우 보수적이 된다.

---

## 3.5 Full Model에서의 Stepwise 적용

본 부록에서 다룬 Stepwise 변수 선택은 [다변량 회귀](../../part4_modeling/multivariate-regression.md)의 Full Model 구성 과정에서 다음과 같이 적용된다.

| 단계 | Stepwise 적용 | 설명 |
|------|-------------|------|
| **① 영역 내 다변량 회귀** | 영역별 Forward/Backward | 재무 변수끼리, CB 변수끼리 각각 Stepwise 수행. 같은 원천 안에서 중복 정보를 가진 변수를 걸러낸다 |
| **② 대표 변수 확정** | 영역별 결과 통합 | 각 영역에서 Stepwise로 살아남은 변수가 해당 영역의 대표 변수 |
| **③ 전체 영역 통합** | 전체 Forward/Backward | 모든 영역의 대표 변수를 한꺼번에 투입해 최종 변수 확정. 영역 간 상관까지 통제 |

Stepwise는 중복 정보를 줄이지만 다중공선성 진단(VIF)과 부호 점검을 대신하지 않는다. 두 가지는 Stepwise 후 별도로 확인한다.

!!! tip "실무 포인트"
    영역 내 Stepwise에서는 Forward Selection이 주로 쓰이고, 전체 통합에서는 Both(Forward + Backward)가 일반적이다. AIC 기준을 적용하면 p = 0.05 기준보다 너그럽게 변수를 받아들이므로(3.4절 참고 박스) 예측력 중심의 선택이 된다. 대신 과적합 점검(OOT 성능)이 더 중요해진다.

---

## 3.6 도구별 Stepwise 동작 비교

같은 "Stepwise"라도 도구마다 Forward/Backward에서 쓰는 기준, 기본값, 종료 조건이 다르다. 그래서 도구를 바꾸면 **같은 데이터에서도 다른 변수가 선택**될 수 있다.

### SAS — PROC LOGISTIC (SELECTION=STEPWISE)

국내 금융권 스코어카드 개발에서 널리 쓰이는 도구다.

**Forward 진입:**

- 모형 밖 각 효과의 **Score 카이제곱**을 계산하고, 그중 가장 큰 값이 SLENTRY 수준에서 유의하면 그 효과를 추가한다. Score 검정은 현재 모형(제외 모형)만으로 계산되므로 후보마다 모형을 새로 적합할 필요가 없다.
- SLENTRY 기본값은 0.05다.

**Backward 제거:**

- 모형 안 각 효과의 **Wald 카이제곱**으로 잔류 여부를 판단하고, SLSTAY 수준에서 유의하지 않은 효과를 제거한다. 이미 적합된 모형의 계수와 분산만으로 계산된다.
- SLSTAY 기본값은 0.05다.

**종료 조건:**

- SAS 문서는 "더 이상 추가할 효과가 없거나, 현재 모형이 **이전에 방문한 모형과 같아지면**" Stepwise를 종료한다고 설명한다. 즉 같은 모형으로 되돌아가는 순환(cycling)이 생기면 그 시점에서 멈춘다.

**부호 제약 / VIF:**

- Stepwise 자체에는 부호 제약이나 VIF 검증이 들어 있지 않다. 분석가가 Stepwise 실행 후 계수 부호와 다중공선성을 별도로 진단해야 한다.

!!! tip "Score Test를 진입에 쓰는 이유"
    Score Test는 현재 모형 1개만 적합하면 모든 후보의 통계량을 한꺼번에 계산할 수 있다. 반면 LR Test는 후보마다 포함 모형을 새로 적합해야 하므로 후보가 100개면 적합이 100번 더 필요하다. 진입 단계는 후보가 많으므로 계산량 차이가 크다.

### R — step() / MASS::stepAIC()

R의 기본 Stepwise 함수는 p-value가 아닌 **정보량 기준(AIC)** 으로 동작한다.

**Forward 진입:**

- 후보 변수를 하나씩 추가한 모형의 AIC를 계산한다.
- AIC가 가장 많이 감소하는 변수를 추가한다. 감소하는 후보가 없으면 종료한다.
- 가설검정이 없으므로 SLENTRY 같은 임계값을 정할 필요가 없다.

**Backward 제거:**

- 현재 모형에서 변수를 하나씩 빼 본 AIC를 계산한다.
- 제거 시 AIC가 감소하는 변수가 있으면 그중 가장 많이 감소하는 변수를 제거한다.

**과적합 방지:**

- 페널티 계수 \(k\)(기본 2)가 변수 수 증가에 대한 제동 역할을 한다. 페널티 계수를 \(\ln n\)으로 바꾸면 BIC 기준이 되어 대표본에서 더 보수적으로 선택한다(R 문서 기준).

**한계:**

- AIC 기준은 "이 변수가 통계적으로 유의한가"에 직접 답하지 않는다. 감독 보고에서 변수별 유의성 근거를 따로 제시해야 한다.
- 계수 부호, VIF, 도메인 제약 등 신용평가 특화 검증이 없다.

### Python — 내장 기능 없음

Python의 대표적인 통계·머신러닝 라이브러리(statsmodels, scikit-learn)에는 p-value 기반 Stepwise 로지스틱 회귀가 **내장되어 있지 않다**.

- statsmodels의 로지스틱 회귀 결과에서 Wald p-value, 로그우도(LR 계산용), AIC/BIC를 얻을 수 있고, 누락 변수에 대한 Score 검정 기능도 제공된다.
- 그러나 Forward/Backward 반복, 진입·제거 판단, 순환 방지 같은 **절차 전체는 사용자가 직접 구성**해야 한다.

!!! warning "Stepwise를 직접 구성할 때 흔한 실수"
    "p-value가 가장 작은 변수를 하나씩 추가"하는 Forward-only 구성이 많다. Backward 제거, 순환 방지, 부호 검증이 빠지면 SAS나 R의 Stepwise와 **같은 결과를 재현할 수 없다**. 진입 검정으로 Score 대신 LR을 쓰는 것도 흔한데, 결과는 대표본에서 대개 같지만 계산량이 후보 수만큼 늘어난다.

---

### 통합 비교표

| 항목 | SAS PROC LOGISTIC | R step() | Python (statsmodels 등) |
|------|-------------------|----------|-------------------|
| **Forward 기준** | Score 카이제곱 | AIC 비교 | 미내장 (Score/LR 직접 구성) |
| **Backward 기준** | Wald 카이제곱 | AIC 비교 | 미내장 (Wald/LR 직접 구성) |
| **진입 기준** | SLENTRY (기본 0.05) | AIC 감소 | 사용자 정의 |
| **제거 기준** | SLSTAY (기본 0.05) | 제거 시 AIC 감소 | 사용자 정의 |
| **순환 처리** | 이전 방문 모형으로 돌아오면 종료 | 매 스텝 AIC가 줄어야 하므로 순환 불가 | 사용자 구현 필요 |
| **부호 제약** | 없음 | 없음 | 사용자 구현 필요 |
| **VIF 검증** | 별도 진단 | 별도 진단 | 사용자 구현 필요 |
| **과적합 방지** | SLENTRY로 간접 | AIC 페널티 내장 | 사용자 구현 필요 |
| **감독기관 보고** | p-value 제공 | AIC 중심 | 구현에 따라 다름 |

!!! note "대표본에서 세 검정의 관계"
    Score, Wald, LR 검정은 표본이 커질수록 점근적으로 같은 분포를 따르므로, 대표본에서는 진입·제거 판정이 대개 일치한다. 다만 통계량의 크기까지 같은 것은 아니며, 효과가 매우 큰 변수에서는 Wald 통계량이 오히려 작아지는 현상(Hauck-Donner, [통계 검정 이론 4.3절](score-wald-lr-test.md))이 있으므로 경계선에 있는 변수는 LR로 확인한다.

!!! warning "SLENTRY vs SLSTAY 설정의 영향"
    SAS 기본값은 SLENTRY = SLSTAY = 0.05로 같다. SLSTAY를 SLENTRY보다 엄격하게(작게) 두면, p = 0.04로 들어온 변수가 다음 스텝에서 p = 0.06이 되는 것만으로 바로 제거되는 등 진입·제거가 반복되기 쉽다. 그래서 실무에서는 SLSTAY ≥ SLENTRY(예: 0.05 / 0.10)로 두는 경우가 많다. 참고로 Bendel & Afifi(1977)는 선형회귀 Forward 선택의 정지 규칙을 비교해, 소표본에서는 0.05보다 느슨한 진입 수준(약 0.15~0.35)이 예측오차 측면에서 더 나았다고 보고했다. 진입 기준을 너무 엄격하게 잡는 것도 능사가 아니라는 근거다.

---

### WoE 스코어카드에서 추가로 점검할 항목

표준 Stepwise(SAS/R)에는 없지만, WoE 기반 스코어카드에서는 다음 점검을 함께 하는 것이 좋다. 아래는 하나의 구성 예이며, 조직의 모형 개발 기준에 맞춰 조정한다.

| 점검 항목 | 내용 |
|-----------|--------|
| **부호 (\(\beta < 0\))** | WoE 변수의 정상 부호는 음수다. \(\beta\)가 양(+)으로 바뀌면 다중공선성·억제효과를 의심하고 원인 변수를 찾는다([다변량 회귀](../../part4_modeling/multivariate-regression.md)) |
| **VIF** | 변수 투입 후 VIF가 급등하면 이미 들어온 변수의 계수까지 불안정해진다. 스텝마다 확인하면 원인 변수를 찾기 쉽다 |
| **변별력 기준 보조 판단** | 대표본에서는 거의 모든 변수가 유의하므로, 유의성은 관문으로 쓰고 실질적인 KS·AR 개선폭으로 우선순위를 본다 |
| **개발/검증 표본 동시 확인** | 개발 표본에서 AR이 올라도 검증 표본에서 떨어지면 과적합 신호다 |
| **종료 판단** | 변수 추가에 따른 AR 개선이 미미해지면 복잡도만 늘어나므로 그 지점에서 멈추는 것을 검토한다 |

!!! note "부호 역전이 여러 변수에서 동시에 생기면"
    부호가 역전된 변수가 여러 개면 하나씩 빼는 순서에 따라 결과가 달라질 수 있다. 역전 변수의 몇 가지 조합을 빼 보며 부호가 정상화되고 변별력 손실이 가장 작은 조합을 고르는 방법을 쓸 수 있다. 어떤 방법이든 선택 근거를 모형 문서에 남긴다.

---

### 참고 문헌 및 공식 문서

**학술 문헌:**

<div class="source-ref">

- Bendel, R.B. & Afifi, A.A. (1977). "Comparison of Stopping Rules in Forward 'Stepwise' Regression." *Journal of the American Statistical Association*, 72(357), 46–53. — Forward 선택의 정지 규칙(유의수준) 비교
- Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X. (2013). *Applied Logistic Regression*. 3rd ed. Wiley. — 로지스틱 회귀 변수 선택 전반
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*. 2nd ed. Springer. — AIC/BIC 기반 모형 선택 이론
- Rao, C.R. (1948). "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Proceedings of the Cambridge Philosophical Society*, 44, 50–57. — Score Test(Lagrange Multiplier Test) 원논문

</div>

**SAS 공식 문서:**

<div class="source-ref">

- SAS Institute. *SAS/STAT User's Guide: The LOGISTIC Procedure*
    - [MODEL Statement — SELECTION=, SLENTRY=, SLSTAY= (기본값 0.05)](https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_logistic_syntax22.htm)
    - [Effect-Selection Methods — Stepwise 종료 조건](https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/statug_logistic_sect029.htm)

</div>

**R 공식 문서:**

<div class="source-ref">

- R Core Team. `step()` — AIC 기반 Stepwise, 페널티 계수 k (기본 2, log(n)이면 BIC)
    - [R Documentation: step](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/step.html)
- Venables, W.N. & Ripley, B.D. (2002). *Modern Applied Statistics with S*. 4th ed. Springer. — `MASS::stepAIC()`
    - [R Documentation: stepAIC](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/stepAIC.html)

</div>

**Python 공식 문서:**

<div class="source-ref">

- statsmodels — 로지스틱 회귀 결과의 p-value, AIC/BIC
    - [statsmodels: Logit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
- statsmodels — 누락 변수에 대한 Score 검정
    - [statsmodels: score_test](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.BinaryResults.score_test.html)

</div>
