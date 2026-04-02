# One-Hot Encoding + No Intercept 회귀

## 2.1 인코딩 방식 — 일반 더미처리 vs One-Hot Encoding

범주형 변수(또는 구간으로 나뉜 연속형 변수)를 회귀모형에 투입하는 방식은 크게 두 가지다. 두 방식의 차이를 먼저 이해해야 No Intercept 방법론의 필요성이 명확해진다.

| 구분 | 일반 더미처리 (k−1 방식) | One-Hot Encoding (k개 + No Intercept) |
|------|----------------------|-------------------------------------|
| **더미 변수 수** | \(k-1\)개 (1개를 기준 Baseline으로 제거) | \(k\)개 (모든 구간에 더미 생성) |
| **절편(Intercept)** | 반드시 포함 (Full-rank 유지를 위해 필수) | 제거 (No Intercept) |
| **다중공선성** | \(k-1\)개 + 절편 → Full-rank | \(k\)개이지만 절편 없음 → Full-rank |
| **\(\hat{\beta}\) 해석** | 기준 구간 대비 *상대적* 로그 오즈 차이 | 해당 구간 자체의 *절대적* 로그 오즈 추정 |
| **WoE와 비교** | 직접 비교 불가 (기준값이 있어 스케일 다름) | 직접 비교 가능 (\(\hat{\beta}_j \approx -\text{WoE}_j\), [증명](beta-woe-proof.md) 참조) |

!!! tip "일반 더미처리(k−1)에서 절편이 반드시 필요한 이유"
    \(k\)개 구간에 대해 \(k\)개의 더미변수를 만들면, **어느 고객이든 반드시 하나의 더미에만 해당**되므로 \(D_1 + D_2 + \cdots + D_k = 1\)이 항상 성립한다. 이는 절편 열(모든 원소가 1인 열)과 **완벽한 선형 종속 관계**를 만들어 설계 행렬(Design Matrix)이 Full-rank를 위반하게 된다.

    수학적으로 이 상태에서는 \((X^TX)^{-1}\)이 존재하지 않아 회귀 계수를 유일하게 추정할 수 없다(Perfect Multicollinearity). 이를 해결하는 방법이 **하나의 구간을 제거하여 \(k-1\)개의 더미만 투입하는 것**이다. 제거된 구간이 Baseline이 되며, 나머지 \(\hat{\beta}_j\)는 Baseline 대비 로그 오즈의 상대적 차이를 의미한다.

!!! tip "k−1 방식의 해석상 한계"
    기준 구간을 Bin1(가장 불량률이 높은 구간)로 설정하면 \(\hat{\beta}_2, \hat{\beta}_3, \ldots\)는 각각 "Bin1에 비해 얼마나 좋은가"를 나타낸다. 분석가 입장에서는 **각 구간 자체의 절대적 부도 위험 수준을 직관적으로 파악하기 어렵고**, WoE와 직접 비교도 불가능하다.

반면 **One-Hot Encoding + No Intercept** 방식은 절편을 제거함으로써 다중공선성 문제를 우회하면서 \(k\)개 더미를 모두 사용한다. 이 경우 각 \(\hat{\beta}_j\)는 해당 구간의 로그 오즈를 직접 추정하므로 WoE와 1:1 비교가 가능하다.

### 설계 행렬 비교 (5개 구간 예시)

<table markdown>
  <thead>
    <tr>
      <th rowspan="2">고객</th>
      <th rowspan="2">Bin</th>
      <th colspan="4" style="text-align:center">일반 더미처리 (k−1 = 4개, Bin1 기준)</th>
      <th colspan="5" style="text-align:center">One-Hot Encoding (k = 5개, No Intercept)</th>
    </tr>
    <tr>
      <th>D<sub>2</sub></th><th>D<sub>3</sub></th><th>D<sub>4</sub></th><th>D<sub>5</sub></th>
      <th>D<sub>1</sub></th><th>D<sub>2</sub></th><th>D<sub>3</sub></th><th>D<sub>4</sub></th><th>D<sub>5</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td>고객 A</td><td>Bin1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
    <tr><td>고객 B</td><td>Bin2</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
    <tr><td>고객 C</td><td>Bin3</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
    <tr><td>고객 D</td><td>Bin4</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr>
    <tr><td>고객 E</td><td>Bin5</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>
  </tbody>
</table>

!!! success "핵심 정리"
    일반 더미처리(\(k-1\) + 절편)와 One-Hot Encoding(No Intercept)은 모형의 **적합도(Log-Likelihood, AIC 등)가 수학적으로 동일**하다. 단, 후자는 각 \(\hat{\beta}_j\)를 WoE와 직접 비교·검증할 수 있다는 해석상 이점 때문에 단변량 로지스틱 회귀에서 채택한다.

## 2.2 No Intercept의 의미

One-Hot Encoding으로 변환된 더미변수들을 독립변수로, 절편 없이 로지스틱 회귀를 수행한다:

$$
\ln\!\left(\frac{p_i}{1-p_i}\right) = \hat{\beta}_1 D_{i1} + \hat{\beta}_2 D_{i2} + \hat{\beta}_3 D_{i3} + \hat{\beta}_4 D_{i4} + \hat{\beta}_5 D_{i5} \tag{1}
$$

!!! note "No Intercept의 효과"
    일반적인 회귀에서 더미변수를 쓸 때는 하나를 기준(Reference)으로 제거하고 나머지 \(k-1\)개만 투입한다(다중공선성 방지). 하지만 **No Intercept 모형에서는 절편이 없으므로 모든 더미변수(\(k\)개)를 투입**할 수 있다. 이렇게 하면 각 \(\hat{\beta}_j\)가 해당 Bin에 속하는 고객의 **log-odds를 직접 추정**하게 된다. 현재 y=1=Bad 설정에서 이 값은 WoE와 **부호가 반대이고 크기가 대응**한다(\(\hat{\beta}_j \approx -\text{WoE}_j\)). 수학적 증명은 [다음 절](beta-woe-proof.md)에서 다룬다.

### 구현 코드

??? example "Python — One-Hot + No Intercept 단변량 로지스틱 회귀"

    ```python
    import pandas as pd
    import statsmodels.api as sm

    # 1) One-Hot 더미 생성 (k개, drop_first=False)
    dummies = pd.get_dummies(df["bin"], prefix="bin", drop_first=False)

    # 2) No Intercept 단변량 로지스틱 회귀
    model = sm.Logit(df["bad_flag"], dummies)
    result = model.fit(disp=False)
    print(result.summary())

    # 3) Bin별 Wald p-value 확인
    print(result.pvalues)

    # 4) LRT: Null Model 대비
    #    Null deviance = -2 * log-likelihood(intercept only)
    null_model = sm.Logit(df["bad_flag"], pd.Series(1, index=df.index))
    null_result = null_model.fit(disp=False)

    lr_stat = -2 * (null_result.llf - result.llf)
    from scipy import stats
    p_value = stats.chi2.sf(lr_stat, df=dummies.shape[1])
    print(f"LRT statistic: {lr_stat:.2f}, p-value: {p_value:.4f}")
    ```

### 실행 결과 예시 (매출액 5구간, Balanced Sample, y=1=Bad)

```
                 coef    std err          z      P>|z|
bin_1          1.8100      0.210      8.619      0.000
bin_2          0.4900      0.180      2.722      0.006
bin_3         -0.0100      0.170     -0.059      0.953
bin_4         -0.8400      0.190     -4.421      0.000
bin_5         -0.9000      0.230     -3.913      0.000

LRT statistic: 142.31, p-value: 0.0000
```

!!! note "결과 해석"
    - `coef` 열이 \(\hat{\beta}\)이며, WoE 값(−1.79, −0.51, 0.00, +0.85, +0.92)과 **부호가 반대이고 크기가 대응**한다 → \(\hat{\beta} \approx -\text{WoE}\)
    - 불량률이 높은 Bin 1(54.5%)에서 β가 가장 크고(+1.81), 불량률이 낮은 Bin 5(7.4%)에서 β가 가장 작다(−0.90) → Bad odds 방향과 일치
    - `P>|z|`가 Wald p-value. Bin 3만 0.953으로 비유의하지만 단조성 유지 구간이므로 허용
    - LRT p-value < 0.001로 변수 전체 유의성 확인
