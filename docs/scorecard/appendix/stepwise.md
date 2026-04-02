# Stepwise 변수 선택: 단계별 동작 원리

Stepwise 변수 선택은 매 스텝마다 Forward(진입) → Backward(제거) 순서로 반복한다. 두 단계에서 서로 다른 검정을 사용하는 이유는 **속도와 정확도의 균형** 때문이다. 검정의 수식과 원리는 [통계 검정 이론](score-wald-lr-test.md)에서 상세히 다룬다.

---

## 4.1 WoE 방식에서의 Stepwise

변수당 계수가 1개(\(\beta_j\))이므로 진입/탈락 판단이 단순하다.

:material-arrow-right-bold: **Forward 진입 — Score Test,** \(\chi^2(1)\)

아직 모형에 없는 후보 변수 \(X_j\)에 대해, "현재 모형의 잔차와 \(X_j\)의 상관관계"를 측정한다. 재적합 없이 계산 가능. 후보 변수 중 Score p-value가 가장 작고 SLENTRY(기본 0.05) 미만인 변수 1개를 추가한다.

:material-arrow-left-bold: **Backward 제거 — Wald Test,** \(\chi^2(1)\)

현재 모형에 포함된 각 변수의 \(\hat{\beta}_j\)를 표준오차로 나눠 유의성을 평가한다. Wald p-value가 가장 크고 SLSTAY(0.1) 초과인 변수 1개를 제거한다.

!!! success "구현 단순성"
    변수당 p-value가 1개뿐이므로 구현이 간단하다. 모든 변수를 개별 컬럼으로 처리하면 된다.

---

## 4.2 더미 방식에서의 Stepwise

변수 X1이 구간 3개(더미 2개: \(\gamma_1\), \(\gamma_2\))라면, 진입/탈락은 **변수(그룹) 단위**로 이루어진다. 개별 더미 단위가 아니다.

:material-arrow-right-bold: **Forward 진입 — 결합 Score Test,** \(\chi^2\)**(구간수−1)**

"\(\gamma_1 = \gamma_2 = 0\)"이라는 결합 제약 하에 X1 그룹 전체의 Score Stat을 계산한다. \(\gamma_1\)이 아무리 유의해 보여도, X1 그룹의 결합 p-value가 SLENTRY를 넘으면 X1은 진입하지 못한다.

:material-arrow-left-bold: **Backward 제거 — 결합 Wald Test,** \(\chi^2\)**(구간수−1)**

\(\hat{\gamma}_1\), \(\hat{\gamma}_2\)의 공분산을 반영한 행렬 연산으로 X1 그룹 전체의 Wald Stat을 계산한다. \(\gamma_2\)가 개별적으로 비유의해도, 결합 Wald p-value가 SLSTAY 이하면 X1 전체가 잔류한다.

---

## 4.3 구간 일부만 비유의일 때의 처리

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

## 4.4 AIC 기반 Stepwise

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

## 4.5 Python 구현 시 주의사항

Python의 statsmodels, sklearn에는 Stepwise와 동일한 기능이 내장되어 있지 않다. 직접 구현할 때 두 방식의 핵심 차이:

| 항목 | WoE 방식 Python 구현 | 더미 방식 Python 구현 |
|------|---------------------|---------------------|
| **변수 처리 단위** | 컬럼 = 변수 (1:1) | 더미 그룹 = 원본 변수 (수동 매핑 필요) |
| **Forward 검정** | LR Test, df=1 고정 | 결합 LR Test, df=더미 개수 (변수마다 다름) |
| **Backward 검정** | `model.pvalues[var]` 한 줄 | \(\gamma^T \cdot V^{-1} \cdot \gamma\) 행렬 연산 직접 구현 |
| **구현 난이도** | 낮음 | 높음 |
| **구현 정확성** | 간단히 동일 결과 | 그룹 정의 + df 관리 정확해야 동일 |

=== "Python — WoE 방식 Stepwise"

    ```python
    # WoE 방식 Stepwise — Forward: LR Test / Backward: Wald Test
    # 변수당 β 1개 → 컬럼 = 변수 (1:1 대응), 구현 단순
    import statsmodels.api as sm
    from scipy import stats

    def stepwise_logistic_woe(X, y, slentry=0.05, slstay=0.10):
        included = []

        while True:
            changed = False

            # ── Forward: LR Test (Score Test 근사) ──────────────────
            excluded = [v for v in X.columns if v not in included]
            best_pval, best_var = slentry, None
            X_base = sm.add_constant(X[included]) if included \
                     else sm.add_constant(X[[]])
            model_base = sm.Logit(y, X_base).fit(disp=0)

            for var in excluded:
                model_new = sm.Logit(
                    y, sm.add_constant(X[included + [var]])
                ).fit(disp=0)
                lr_stat = 2 * (model_new.llf - model_base.llf)
                pval = stats.chi2.sf(lr_stat, df=1)
                if pval < best_pval:
                    best_pval, best_var = pval, var
            if best_var:
                included.append(best_var); changed = True

            # ── Backward: Wald Test ──────────────────────────────────
            if included:
                model_cur = sm.Logit(
                    y, sm.add_constant(X[included])
                ).fit(disp=0)
                pvals = model_cur.pvalues[included]
                worst = pvals.idxmax()
                if pvals[worst] > slstay:
                    included.remove(worst); changed = True

            if not changed: break

        return sm.Logit(y, sm.add_constant(X[included])).fit(), included
    ```

=== "Python — 더미 방식 Stepwise"

    ```python
    # 더미 방식 Stepwise — Forward: 결합 Score Test(LR 근사) / Backward: 결합 Wald Test
    # 핵심: 더미 그룹을 원본 변수 단위로 묶어서 검정
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats

    dummy_groups = {
        'R1_DSR':       ['R1_DSR_1',       'R1_DSR_2'],
        'R1_CF1200901': ['R1_CF1200901_1'],
        'R1_업력':       ['R1_업력_1',        'R1_업력_2'],
    }

    def group_lr_pvalue(y, X_base_cols, new_dummies, X_all, model_base):
        X_new = sm.add_constant(X_all[X_base_cols + new_dummies])
        model_new = sm.Logit(y, X_new).fit(disp=0)
        lr_stat = 2 * (model_new.llf - model_base.llf)
        df = len(new_dummies)
        return stats.chi2.sf(lr_stat, df), model_new

    def group_wald_pvalue(model, dummy_cols):
        gamma = model.params[dummy_cols].values
        V = model.cov_params().loc[dummy_cols, dummy_cols].values
        wald_stat = gamma @ np.linalg.inv(V) @ gamma
        df = len(dummy_cols)
        return stats.chi2.sf(wald_stat, df)

    def stepwise_logistic_dummy(X_dummies, y, dummy_groups,
                                 slentry=0.05, slstay=0.10):
        included_vars = []
        included_cols = []

        while True:
            changed = False

            X_base = sm.add_constant(X_dummies[included_cols]) \
                     if included_cols \
                     else sm.add_constant(X_dummies[[]])
            model_base = sm.Logit(y, X_base).fit(disp=0)

            excluded_vars = [v for v in dummy_groups
                             if v not in included_vars]
            best_pval, best_var = slentry, None

            for var in excluded_vars:
                dummies = dummy_groups[var]
                pval, _ = group_lr_pvalue(
                    y, included_cols, dummies, X_dummies, model_base
                )
                if pval < best_pval:
                    best_pval, best_var = pval, var

            if best_var:
                included_vars.append(best_var)
                included_cols += dummy_groups[best_var]
                changed = True

            if included_cols:
                model_cur = sm.Logit(y, sm.add_constant(
                    X_dummies[included_cols])).fit(disp=0)
                worst_var, worst_pval = None, 0

                for var in included_vars:
                    dummies = dummy_groups[var]
                    pval = group_wald_pvalue(model_cur, dummies)
                    if pval > worst_pval:
                        worst_pval, worst_var = pval, var

                if worst_pval > slstay:
                    included_vars.remove(worst_var)
                    for col in dummy_groups[worst_var]:
                        included_cols.remove(col)
                    changed = True

            if not changed: break

        final_model = sm.Logit(y, sm.add_constant(
            X_dummies[included_cols])).fit()
        return final_model, included_vars
    ```

!!! warning "더미 방식에서 WoE 코드를 그대로 쓰면 안 되는 이유"
    더미를 단순히 컬럼으로 펼쳐놓으면 `R1_DSR_1`, `R1_DSR_2`가 각각 별개 변수로 처리된다. Forward에서 `R1_DSR_1`만 진입하고 `R1_DSR_2`는 탈락하는 등 원본 변수 단위가 아닌 더미 단위로 쪼개지는 문제가 생긴다. 변수-더미 매핑을 직접 관리하여 **그룹 단위 검정**이 이루어지도록 구현해야 한다.

### 더미 방식 구현이 까다로운 이유 — 핵심 차이 3가지

<div class="review-item" markdown>
  <span class="review-badge">1</span>
  <span class="review-title">변수-더미 매핑 테이블을 수동으로 관리해야 한다</span>
</div>

WoE 방식은 컬럼 = 변수가 1:1이므로 `X.columns`를 그대로 순회하면 된다. 더미 방식은 `R1_DSR → [R1_DSR_1, R1_DSR_2]`처럼 원본 변수와 더미 컬럼의 매핑을 사전에 정의하고 관리해야 한다.

<div class="review-item" markdown>
  <span class="review-badge">2</span>
  <span class="review-title">Forward LR Test의 df가 변수마다 다르다</span>
</div>

WoE 방식은 모든 변수가 df=1로 고정이다. 더미 방식은 구간 수에 따라 df가 다르다. 2구간 변수는 df=1, 3구간은 df=2, 5구간은 df=4. df가 다르면 같은 LR Stat이라도 p-value가 달라지므로, 각 변수의 더미 개수를 정확히 파악해야 한다.

<div class="review-item" markdown>
  <span class="review-badge">3</span>
  <span class="review-title">Backward Wald Test가 행렬 연산이다</span>
</div>

WoE 방식은 `model.pvalues[var]` 한 줄로 끝난다. 더미 방식은 \(\gamma\) 벡터와 분산-공분산 행렬을 꺼내서 \(\gamma^T \cdot V^{-1} \cdot \gamma\) 행렬 연산을 직접 수행해야 한다. 더미 간 공분산(상관관계)까지 반영해야 결합 검정이 정확하기 때문이다.

---

## 4.6 Full Model에서의 Stepwise 적용

본 부록에서 다룬 Stepwise 변수 선택은 [모델링 · 스코어카드](../part4_modeling/multivariate-regression.md)의 Full Model 구성 과정에서 다음과 같이 적용된다.

| 단계 | Stepwise 적용 | 설명 |
|------|-------------|------|
| **① 영역 내 다변량 회귀** | 영역별 Forward/Backward | 재무 변수끼리, CB 변수끼리 각각 Stepwise 수행. 동일 원천 내 다중공선성 제거 |
| **② 대표 변수 확정** | 영역별 결과 통합 | 각 영역에서 Stepwise로 살아남은 변수가 해당 영역의 대표 변수 |
| **③ 전체 영역 통합** | 전체 Forward/Backward | 모든 영역의 대표 변수를 한꺼번에 투입하여 최종 변수 확정. 영역 간 상관관계까지 통제 |

!!! tip "실무 포인트"
    영역 내 Stepwise에서는 Forward Selection이 주로 사용되고, 전체 통합에서는 Both(Forward + Backward)가 일반적이다. AIC 기준(본 부록 참고)을 적용하면 p-value 기반보다 예측력 중심의 변수 선택이 가능하다.
