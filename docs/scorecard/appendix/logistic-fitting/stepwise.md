# Stepwise 변수 선택: 단계별 동작 원리

Stepwise 변수 선택은 매 스텝마다 Forward(진입) → Backward(제거) 순서로 반복한다. 두 단계에서 서로 다른 검정을 사용하는 이유는 **속도와 정확도의 균형** 때문이다. 검정의 수식과 원리는 [통계 검정 이론](score-wald-lr-test.md)에서 상세히 다룬다.

---

## 3.1 WoE 방식에서의 Stepwise

변수당 계수가 1개(\(\beta_j\))이므로 진입/탈락 판단이 단순하다.

**→** **Forward 진입 — Score Test,** \(\chi^2(1)\)

아직 모형에 없는 후보 변수 \(X_j\)에 대해, "현재 모형의 잔차와 \(X_j\)의 상관관계"를 측정한다. 재적합 없이 계산 가능. 후보 변수 중 Score p-value가 가장 작고 SLENTRY(기본 0.05) 미만인 변수 1개를 추가한다.

**←** **Backward 제거 — Wald Test,** \(\chi^2(1)\)

현재 모형에 포함된 각 변수의 \(\hat{\beta}_j\)를 표준오차로 나눠 유의성을 평가한다. Wald p-value가 가장 크고 SLSTAY(0.1) 초과인 변수 1개를 제거한다.

!!! success "구현 단순성"
    변수당 p-value가 1개뿐이므로 구현이 간단하다. 모든 변수를 개별 컬럼으로 처리하면 된다.

---

## 3.2 더미 방식에서의 Stepwise

변수 X1이 구간 3개(더미 2개: \(\gamma_1\), \(\gamma_2\))라면, 진입/탈락은 **변수(그룹) 단위**로 이루어진다. 개별 더미 단위가 아니다.

**→** **Forward 진입 — 결합 Score Test,** \(\chi^2\)**(구간수−1)**

"\(\gamma_1 = \gamma_2 = 0\)"이라는 결합 제약 하에 X1 그룹 전체의 Score Stat을 계산한다. \(\gamma_1\)이 아무리 유의해 보여도, X1 그룹의 결합 p-value가 SLENTRY를 넘으면 X1은 진입하지 못한다.

**←** **Backward 제거 — 결합 Wald Test,** \(\chi^2\)**(구간수−1)**

\(\hat{\gamma}_1\), \(\hat{\gamma}_2\)의 공분산을 반영한 행렬 연산으로 X1 그룹 전체의 Wald Stat을 계산한다. \(\gamma_2\)가 개별적으로 비유의해도, 결합 Wald p-value가 SLSTAY 이하면 X1 전체가 잔류한다.

---

## 3.3 구간 일부만 비유의일 때의 처리

더미 방식의 실무적 핵심 이슈다. Stepwise가 X1을 잔류시켰더라도, 개별 더미 계수를 확인해야 한다.

!!! note "Case 1 — 결합 검정 유의, 그러나 \(\gamma_2\) 개별 p > 0.1"
    결합 검정이 유의하므로 X1은 잔류한다. 그러나 "\(\gamma_2\)가 비유의하다 = 구간1과 구간2를 합쳐도 된다"는 신호다. **Stepwise 후 구간 재검토(Re-classing)를 수행**하고, 구간1+2를 합친 뒤 모형 재적합을 권고한다.

!!! success "Case 2 — 결합 검정도 비유의"
    X1 전체가 제거된다. 깔끔하다. 변수 선별 단계(IV, 단변량 로지스틱 회귀)를 재검토할 필요가 있다.

!!! warning "Case 3 — 결합 검정 비유의, 그러나 \(\gamma_1\) 개별 p < 0.05"
    X1이 제거된다. \(\gamma_2\) 더미가 노이즈를 흡수해 결합 검정을 희석시킨 것이다. **구간화 재설계가 필요하다.** \(\gamma_2\) 구간을 기준 구간과 합쳐 더미 수를 줄이면 결합 검정이 살아날 수 있다.

!!! danger "결론"
    **더미 방식에서 Stepwise는 변수 선택의 끝이 아니라 구간 재검토의 출발점이다.** 최종 모형 확정 전 반드시 각 변수의 더미 계수 개별 유의성과 부호 방향을 점검해야 한다.

---

## 3.4 AIC 기반 Stepwise

p-value 방식이 "이 변수의 계수가 통계적으로 0과 다른가?"를 묻는다면, AIC 방식은 **"이 변수가 있는 모형이 없는 모형보다 전체적으로 더 좋은가?"**를 묻는다. 가설검정(p-value)이 없고 두 숫자의 크기만 비교한다.

$$
AIC = -2\log L + 2k \tag{A.7}
$$

\(-2\log L\)은 모형의 설명력(작을수록 좋음), \(2k\)는 변수 개수에 대한 패널티다. 변수를 추가하면 \(\log L\)은 항상 올라가지만 패널티도 늘어나므로, **설명력 향상이 패널티를 상쇄할 만큼 충분한가**로 진입/탈락을 결정한다.

### Forward 예시 — 변수 추가

현재 모형에 변수 3개(DSR, 카드이용률, 업력)가 있고, 후보로 부채비율을 고려하는 상황:

!!! note ""
    현재 모형 (k=3): AIC = −2 × (−320.5) + 2 × 3 = **647.0**

    후보 모형 (k=4): AIC = −2 × (−318.1) + 2 × 4 = **644.2**

    644.2 < 647.0 → 부채비율 **추가**

### Backward 예시 — 변수 제거

현재 모형에 변수 4개(DSR, 카드이용률, 업력, 부채비율)가 있고, 각각 제거해보는 상황:

!!! note ""
    현재 모형 (k=4): AIC = **644.2**

    - DSR 제거 (k=3): AIC = 651.3 → 나빠짐
    - 카드이용률 제거 (k=3): AIC = 648.7 → 나빠짐
    - 업력 제거 (k=3): AIC = 646.1 → 나빠짐
    - 부채비율 제거 (k=3): AIC = 643.8 → **좋아짐**

    643.8 < 644.2 → 부채비율 **제거**

Forward에서 들어온 변수라도 다른 변수들과 함께 있을 때 중복 정보가 많으면 Backward에서 걸러진다.

### p-value vs AIC 비교

| 비교 항목 | p-value 기반 | AIC 기반 |
|-----------|------------------------|------------------------|
| **판단 기준** | 유의수준 임계값 (SLENTRY, SLSTAY) | AIC 증감 여부 |
| **가설검정** | 있음 (Score / Wald Test) | 없음 (숫자 비교만) |
| **임계값 설정** | 사람이 직접 설정 필요 | 자동 (AIC 기준) |
| **과적합 방지** | 유의수준으로 간접 통제 | 패널티 항(2k)에 자동 내포 |
| **계수 유의성 설명** | 직관적 (p-value로 설명) | 덜 직관적 |
| **감독기관 보고** | 선호 (금융권 관행) | 상대적으로 덜 사용 |
| **주요 사용처** | 국내 은행 실무 (SAS 등) | 통계/학술 분야 (R 등) |

!!! warning "BIC (Bayesian Information Criterion)"
    AIC의 변형으로 패널티를 더 강하게 준다.

    $$BIC = -2\log L + k \cdot \ln(n)$$

    표본 수(\(n\))가 클수록 패널티가 커지므로 변수를 더 보수적으로 선택한다. 학술 논문이나 계량경제 모형에서 선호된다.

---

## 3.5 Full Model에서의 Stepwise 적용

본 부록에서 다룬 Stepwise 변수 선택은 [모델링 · 스코어카드](../../part4_modeling/multivariate-regression.md)의 Full Model 구성 과정에서 다음과 같이 적용된다.

| 단계 | Stepwise 적용 | 설명 |
|------|-------------|------|
| **① 영역 내 다변량 회귀** | 영역별 Forward/Backward | 재무 변수끼리, CB 변수끼리 각각 Stepwise 수행. 동일 원천 내 다중공선성 제거 |
| **② 대표 변수 확정** | 영역별 결과 통합 | 각 영역에서 Stepwise로 살아남은 변수가 해당 영역의 대표 변수 |
| **③ 전체 영역 통합** | 전체 Forward/Backward | 모든 영역의 대표 변수를 한꺼번에 투입하여 최종 변수 확정. 영역 간 상관관계까지 통제 |

!!! tip "실무 포인트"
    영역 내 Stepwise에서는 Forward Selection이 주로 사용되고, 전체 통합에서는 Both(Forward + Backward)가 일반적이다. AIC 기준(본 부록 참고)을 적용하면 p-value 기반보다 예측력 중심의 변수 선택이 가능하다.

---

## 3.6 도구별 Stepwise 동작 비교

Stepwise 변수 선택의 세부 동작은 도구마다 다르다. 같은 "Stepwise"라 해도 Forward/Backward에서 사용하는 검정, 기본 파라미터, 종료 조건이 모두 다르므로, 도구를 바꾸면 **동일한 데이터에서도 다른 변수가 선택**될 수 있다.

### SAS — PROC LOGISTIC (SELECTION=STEPWISE)

국내 금융권 스코어카드 개발에서 가장 널리 사용되는 도구다.

**Forward 진입:**

- **Score Test**(Lagrange Multiplier)를 기본으로 사용한다. 현재 모형(reduced model)만으로 계산 가능하여 후보 변수마다 full model을 적합할 필요가 없다. 이것이 SAS가 Score Test를 채택한 핵심 이유다.
- Score statistic이 가장 크고, p-value가 **SLENTRY**(기본 0.05) 미만인 변수 1개를 투입한다.
- 동률 시 Score statistic이 더 큰 변수를 선택한다 (순수 통계적 기준).

**Backward 제거:**

- **Wald Test**를 기본으로 사용한다. 이미 적합된 모형에서 각 계수의 \(\hat{\beta}/SE(\hat{\beta})\)를 계산하므로 추가 적합이 불필요하다.
- Wald p-value가 가장 크고, **SLSTAY**(기본 0.05) 초과인 변수 1개를 제거한다.
- SAS 기본값은 SLENTRY = SLSTAY = 0.05로 동일하다.

**Cycling 방지:**

- Backward에서 제거된 변수는 이후 Forward에서 **재진입이 불가**하다. 이 규칙이 없으면 A 진입 → B 제거 → A 재진입 → B 재제거의 무한 반복이 발생할 수 있다.

**부호 제약 / VIF:**

- SAS의 Stepwise 자체에는 부호 제약이나 VIF 검증이 **내장되어 있지 않다**. 분석가가 Stepwise 실행 후 별도로 계수 부호와 다중공선성을 진단해야 한다.

!!! tip "SAS의 Score Test 선택 배경"
    1990년대 SAS 개발 당시 컴퓨팅 자원이 제한적이었다. Score Test는 reduced model 1개만 적합하면 모든 후보의 검정통계량을 동시에 계산할 수 있어 **O(1) 적합**이다. 반면 LRT는 후보 변수마다 full model을 적합해야 하므로 **O(후보 수) 적합**이 필요하다. 현대 하드웨어에서는 이 차이가 미미하지만, SAS의 기본 설정은 그대로 유지되고 있다.

### R — step() / stepAIC()

R은 p-value가 아닌 **정보량 기준(AIC)**으로 Stepwise를 수행한다.

**Forward 진입:**

- 후보 변수를 하나씩 추가한 모형의 AIC를 계산한다.
- AIC가 가장 많이 감소하는 변수를 투입한다. AIC가 감소하지 않으면 종료.
- 가설검정(p-value)이 없으므로 SLENTRY 같은 임계값 설정이 불필요하다.

**Backward 제거:**

- 현재 모형에서 변수를 하나씩 빼본 AIC를 계산한다.
- 제거 시 AIC가 감소하는 변수가 있으면 그 중 가장 많이 감소하는 변수를 제거한다.

**과적합 방지:**

- AIC = \(-2\log L + 2k\)에서 \(2k\) 패널티가 변수 수 증가에 대한 자동 제동 역할을 한다. 변수를 추가했을 때 log-likelihood 개선이 패널티를 상쇄하지 못하면 자동으로 투입하지 않는다.
- BIC = \(-2\log L + k \cdot \ln(n)\)을 사용하면 대표본에서 패널티가 더 강해져 보수적인 변수 선택이 이루어진다.

**한계:**

- AIC 기준은 "이 변수가 통계적으로 유의한가"에 대한 답을 주지 않는다. 금융 감독기관 보고 시 변수별 유의성 근거를 제시하기 어렵다.
- 계수 부호, VIF, 도메인 제약 등 신용평가 특화 검증이 없다.

### Python (statsmodels) — 기본 기능

Python의 statsmodels, scikit-learn에는 **Stepwise가 내장되어 있지 않다**. 직접 구현해야 한다.

- statsmodels의 `Logit.fit()`으로 모형을 적합하면 Wald p-value, LRT, AIC 등을 모두 추출할 수 있다.
- Score Test는 `model.score_test(exog_extra=X_new)`로 계산 가능하다.
- 그러나 Forward/Backward 반복, 변수 진입/제거 판단, Cycling 방지 등의 **오케스트레이션은 사용자가 직접 작성**해야 한다.

!!! warning "Python에서 Stepwise를 구현할 때 흔한 실수"
    단순히 "p-value가 가장 작은 변수를 하나씩 추가"하는 Forward-only 구현이 많다. Backward 제거, Cycling 방지, 부호 검증 등이 빠지면 SAS나 R의 Stepwise와 **동일한 결과를 재현할 수 없다**.

---

### 통합 비교표

| 항목 | SAS PROC LOGISTIC | R step() | Python statsmodels |
|------|-------------------|----------|-------------------|
| **Forward 검정** | Score Test | AIC 비교 | 미내장 (LRT/Score 수동) |
| **Backward 검정** | Wald Test | AIC 비교 | 미내장 (Wald/LRT 수동) |
| **진입 기준** | SLENTRY=0.05 | AIC 감소 | 사용자 정의 |
| **제거 기준** | SLSTAY=0.05 | AIC 증가 | 사용자 정의 |
| **동률 해소** | Score stat 최대 | AIC 최소 | 사용자 정의 |
| **Cycling 방지** | 제거 후 재진입 불가 | AIC 단조성 | 사용자 구현 필요 |
| **부호 제약** | 없음 | 없음 | 사용자 구현 필요 |
| **VIF 검증** | 별도 진단 | 별도 진단 | 사용자 구현 필요 |
| **과적합 방지** | SLENTRY로 간접 | AIC 패널티 내장 | 사용자 구현 필요 |
| **감독기관 보고** | p-value 제공 (선호) | AIC만 (비선호) | 구현에 따라 다름 |

!!! note "대표본에서의 동치"
    관측치가 수십만 건 이상인 대표본에서는 Score Test ≈ LRT ≈ Wald 세 검정이 점근적으로 동치(asymptotically equivalent)다. 따라서 SAS의 Score Test 기반 결과와 LRT 기반 결과는 실질적으로 동일하다.

!!! warning "SLENTRY vs SLSTAY 설정의 영향"
    SAS 기본값은 SLENTRY = SLSTAY = 0.05로 동일하다. Bendel & Afifi (1977)는 SLSTAY ≥ SLENTRY를 권고했다 — SLSTAY가 SLENTRY보다 타이트하면 변수가 진입/제거를 반복하는 cycling이 발생할 수 있기 때문이다. 실무에서는 SLENTRY=0.05, SLSTAY=0.10 조합이 자주 사용된다.

---

### WoE 스코어카드에서의 확장

표준 Stepwise(SAS/R)에는 없지만, WoE 기반 스코어카드에서는 다음 검증이 추가로 필요하다:

| 검증 항목 | 필요성 | 비고 |
|-----------|--------|------|
| **부호 제약 (\(\beta < 0\))** | WoE 방향과 β 부호가 일치해야 단조성 유지 | β ≥ 0은 다중공선성 신호 → 해당 변수 또는 원인 변수 제거 |
| **VIF 실시간 검증** | 매 step 변수 투입 후 다중공선성 즉시 확인 | SAS에서는 Stepwise 후 별도 진단하지만, 변수가 투입된 후 VIF가 급등하면 이미 다른 변수에 영향 |
| **AR/KS 기반 변수 선택** | p-value는 대표본에서 거의 모든 변수가 유의 → 실질적 변별력(AR)로 우선순위 결정 | "통계적 유의성은 관문, 실질적 성능이 선택 기준" |
| **교차검증 로깅** | train에서 AR이 올라도 valid에서 떨어지면 과적합 신호 | 매 step train/valid/test 성능을 동시 기록하여 과적합 조기 감지 |
| **Elbow Criterion** | AR 개선이 미미한 변수를 계속 투입하면 모형 복잡도만 증가 | 첫 추가 변수 delta AR의 일정 비율 이하로 떨어지면 종료 |

!!! note "부호 역전 처리 — Power Set 비교"
    β ≥ 0인 변수가 발생하면 단순히 해당 변수를 제거하는 것이 아니라, 역전 변수들의 **모든 부분집합(power set)**을 비교하여 AR이 가장 높고 부호가 해소되는 조합을 채택한다. 역전 변수가 \(k\)개일 때 \(2^k - 1\)개 조합을 비교하지만, 실무에서 \(k \leq 3\)이므로 최대 7개 비교로 계산 부담이 없다.

---

### 참고 문헌 및 공식 문서

**학술 문헌:**

<div class="source-ref">

- Bendel, R.B. & Afifi, A.A. (1977). "Comparison of Stopping Rules in Forward Stepwise Regression." *Journal of the American Statistical Association*, 72(357), 46–53. — SLENTRY/SLSTAY 설정 근거, cycling 방지 조건
- Hosmer, D.W., Lemeshow, S. & Sturdivant, R.X. (2013). *Applied Logistic Regression*. 3rd ed. Wiley. — 로지스틱 회귀 Stepwise 변수 선택 전반, 도메인 기반 부호 제약 권고
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*. 2nd ed. Springer. — AIC/BIC 기반 모형 선택 이론
- Rao, C.R. (1948). "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Proceedings of the Cambridge Philosophical Society*, 44, 50–57. — Score Test(Lagrange Multiplier Test) 원논문

</div>

**SAS 공식 문서:**

<div class="source-ref">

- SAS Institute. *SAS/STAT User's Guide: The LOGISTIC Procedure* — `SELECTION=STEPWISE` 옵션, `SLENTRY`, `SLSTAY` 파라미터, Score/Wald/LRT 검정 선택
    - [PROC LOGISTIC: Model Statement — SELECTION Option](https://documentation.sas.com/doc/en/statug/15.2/statug_logistic_details06.htm)
    - [PROC LOGISTIC: Details — Variable Selection Methods](https://documentation.sas.com/doc/en/statug/15.2/statug_logistic_details07.htm)

</div>

**R 공식 문서:**

<div class="source-ref">

- R Core Team. `step()` — `stats::step(object, direction="both")`, AIC 기반 양방향 변수 선택
    - [R Documentation: step](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/step.html)
- Venables, W.N. & Ripley, B.D. (2002). *Modern Applied Statistics with S*. 4th ed. Springer. — `MASS::stepAIC()` 함수, BIC 옵션(`k=log(n)`)
    - [R Documentation: stepAIC](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/stepAIC.html)

</div>

**Python 공식 문서:**

<div class="source-ref">

- statsmodels. `discrete.discrete_model.Logit` — `fit()`, `pvalues`, `aic`, `bic` 속성
    - [statsmodels: Logit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
- statsmodels. `LogitResults.score_test()` — Score Test(Lagrange Multiplier) 계산
    - [statsmodels: score_test](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.BinaryResults.score_test.html)
- statsmodels에는 Stepwise 기능이 **미내장**. Forward/Backward 오케스트레이션은 직접 구현 필요.

</div>

---
