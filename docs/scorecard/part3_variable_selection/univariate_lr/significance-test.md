# 유의성 검정

## 4.1 Wald Test — Bin별 유의성

각 Bin의 \(\hat{\beta}_j\)가 0과 통계적으로 유의미하게 다른지 검정한다.

!!! note "Wald Test 귀무가설과 대립가설"
    - **귀무가설 \(H_0\):** \(\beta_j = 0\) — 해당 구간의 로그 오즈가 0이다 (불량과 우량이 1:1 비율)
    - **대립가설 \(H_1\):** \(\beta_j \neq 0\) — 해당 구간의 로그 오즈가 0이 아니다 (구별력이 있다)
    - **목적:** 개별 Bin(구간) 하나하나에 대해 독립적으로 검정. 각 더미 계수가 *개별적으로* 유의한지 확인한다.

$$
W_j = \left(\frac{\hat{\beta}_j}{\widehat{SE}(\hat{\beta}_j)}\right)^2 \sim \chi^2(1) \quad \text{under } H_0: \beta_j = 0 \tag{5}
$$

| Bin | \(\hat{\beta}_j\) | \(\widehat{SE}\) | Wald 통계량 | p-value | 판단 |
|-----|-----|-----|-----|---------|------|
| 1 (1억미만) | +1.81 | 0.21 | 74.2 | <0.001 | 유의 |
| 2 (1억~5억) | +0.49 | 0.18 | 7.4 | 0.007 | 유의 |
| 3 (5억~10억) | −0.01 | 0.17 | 0.003 | 0.954 | 주의 — 아래 해설 참조 |
| 4 (10억~50억) | −0.84 | 0.19 | 19.6 | <0.001 | 유의 |
| 5 (50억초과) | −0.90 | 0.23 | 15.3 | <0.001 | 유의 |

!!! warning "Bin3 비유의 해석"
    Bin3의 WoE = 0.00이고 p-value가 높다는 이유만으로 "합병을 권고"하는 것은 재검토가 필요하다.

    **Wald Test의 귀무가설(\(H_0: \beta_j = 0\))이 의미하는 것:**
    "\(\beta_j = 0\)"은 "해당 구간의 Bad/Good 비율이 1:1로 동일하다"는 의미다. WoE는 이와 달리 "모집단 평균 대비" 비율이므로, WoE = 0은 "\(\beta_j = \ln(N_B/N_G) \neq 0\)"을 뜻하는 게 일반적이다.

    **Classing의 본질적 특성:**
    5개 구간으로 나뉠 때, 가운데 구간(Bin3)은 자연스럽게 모집단 평균에 가까운 Bad Rate를 갖게 된다. 가운데 구간은 \(\ln(n_{B,i}/n_{G,i})\)가 0에 수렴하는 것이 오히려 **정상적인 Classing의 결과**다.

    **단조성(Monotonicity) 기준의 우선순위:**
    Bin3의 WoE가 \((-)\)에서 \((+)\)로 이어지는 단조 흐름 상에 0으로 위치한다면, 이는 Classing이 잘 된 것을 의미한다. WoE 순서가 **\((-) \to 0 \approx \to (+)\)**로 단조성을 유지하는 한, 중간 구간의 p-value 수치만을 근거로 합병하는 것은 오히려 모형의 정보량(IV)을 훼손하는 결과를 낳는다.

    **결론:** Bin3의 비유의는 "이 구간이 구별력을 잃었다"는 게 아니라, "이 구간의 Bad Rate가 모집단 전체 평균과 유사하다"는 사실을 통계적으로 재확인한 것이다. 단조성이 유지되고 변수 전체 LRT가 유의하다면, 합병 없이 Bin3를 유지하는 것이 더 바람직한 판단이다.

---

## 4.2 Likelihood Ratio Test — 변수 전체 유의성

변수 전체가 모형에 기여하는지를 검정한다. Null Model(절편만 있는 모형)과 Full Model(해당 변수 포함)의 Log-Likelihood를 비교한다.

!!! note "LRT 귀무가설과 대립가설"
    - **귀무가설 \(H_0\):** \(\beta_1 = \beta_2 = \cdots = \beta_k = 0\) — 해당 변수의 모든 구간 계수가 동시에 0이다
    - **대립가설 \(H_1\):** 최소한 하나의 \(\beta_j \neq 0\) — 이 변수가 부도 예측에 통계적으로 의미 있다
    - **목적:** 개별 Bin이 아닌 *변수 전체*의 기여도를 평가

$$
\text{LR} = -2[\ell(\text{Null}) - \ell(\text{Full})] \sim \chi^2(df) \tag{6}
$$

\(df\): 추가된 파라미터 수 (No Intercept 모형의 경우 Bin 수와 동일)

!!! tip "LRT 구체 적용 예시 — 3개 구간을 가진 변수의 경우"
    예를 들어 "업력" 변수를 3개 구간(Bin1: 3년 미만, Bin2: 3~10년, Bin3: 10년 초과)으로 나누었다고 하자.

    1. **Null Model:** 업력 변수를 아예 포함하지 않은 모형을 피팅 → \(\ell(\text{Null})\) 산출
    2. **Full Model:** 업력 변수의 3개 더미(\(D_1, D_2, D_3\))를 모두 포함한 모형을 피팅 → \(\ell(\text{Full})\) 산출
    3. **검정:** \(\text{LR} = -2[\ell(\text{Null}) - \ell(\text{Full})]\)을 자유도 \(df = 3\)인 카이제곱 분포와 비교

    LR 통계량이 \(\chi^2(3)\)의 임계값(p=0.05 기준 약 7.81)을 초과하면 \(H_0\)를 기각하고 업력 변수가 모형에 통계적으로 기여한다고 판단한다. **개별 Bin의 Wald p-value가 일부 높더라도 LRT가 유의하면 변수 자체는 모형에 투입할 수 있다.**

### LRT 구현 코드

??? example "Python — LRT 통계량 산출"

    ```python
    import statsmodels.api as sm
    from scipy import stats

    # Full Model: 해당 변수의 k개 더미 투입 (No Intercept)
    full_model = sm.Logit(y, dummies)
    full_result = full_model.fit(disp=False)

    # Null Model: 절편만 포함 (변수 미투입)
    null_model = sm.Logit(y, sm.add_constant(pd.Series(1, index=y.index)))
    null_result = null_model.fit(disp=False)

    # LRT 통계량 산출
    lr_stat = -2 * (null_result.llf - full_result.llf)
    df_diff = dummies.shape[1] - 1   # Full df - Null df
    p_value = stats.chi2.sf(lr_stat, df=df_diff)

    print(f"LRT: {lr_stat:.2f}, df={df_diff}, p={p_value:.6f}")
    ```

!!! warning "Null Model 정의 주의"
    No Intercept 모형에서 LRT의 Null Model은 **절편만 포함한 모형**(변수 미투입)이다. "절편도 없는 빈 모형"이 아니다. Null Model의 절편은 전체 모집단의 log-odds \(\ln(N_B/N_G)\)를 추정하며, LRT는 "변수를 추가함으로써 이 기본 수준 대비 얼마나 설명력이 향상되었는가"를 검정한다.

### 검정 방법 비교

| 검정 | 비교 대상 | 귀무가설 | 언제 사용 | 특징 |
|------|----------|---------|---------|------|
| **Wald Test** | 개별 \(\hat{\beta}\)의 SE 기반 | \(\beta_j = 0\) (개별 Bin) | Bin별 유의성 (Classing 진단) | 빠르고 간편. 대용량 샘플에서 신뢰도 높음 |
| **LRT** | Null vs Full Log-Likelihood | \(\beta_1 = \cdots = \beta_k = 0\) (변수 전체) | 변수 전체 유의성 | Wald보다 정확. 소용량 샘플에서 권장 |
| **Score Test** | Null 모형에서의 Score 함수 | Null 모형 기반 동일 | LRT 대안 (계산 효율) | LRT와 점근적으로 동일. 계산이 더 가벼움 |

!!! note "실무 기준값"
    - Bin별 Wald p-value: < 0.05 (단조성 유지 구간은 관대하게 < 0.10 적용 가능)
    - 변수 전체 LRT p-value: < 0.05
    - 소용량 샘플(Bad < 500건): Wald보다 LRT를 우선 사용
    - p-value는 가이드라인일 뿐. 업무 중요도·IV·단조성과 함께 종합 판단
    - **특히:** 개별 Bin의 Wald p-value ≥ 0.05이더라도, 단조성이 유지되고 LRT가 유의하면 합병 없이 유지 가능

!!! example "실무 참고: 단변량 로지스틱 회귀 유의성 기준"
    전통 스코어카드 개발 실무에서는 단변량 로지스틱 회귀의 유의성 기준을 다음과 같이 적용하는 것이 일반적이다:

    - **변수 전체 LRT p-value < 0.05**를 1차 스크리닝 기준으로 사용
    - Bin별 Wald p-value는 **0.05 ~ 0.10 사이**를 허용 범위로 두되, 단조성 유지 여부를 우선 고려
    - 개별 변수의 통계적 유의성과 함께 **업무적 설명력**을 종합 판단

---

## 4.3 단변량 로지스틱 회귀 결과 해석 매트릭스

| \(\hat{\beta}\) 방향 | p-value | 판단 | 조치 |
|-----|---------|------|------|
| WoE와 부호 반대 (\(\hat{\beta}>0 \leftrightarrow \text{WoE}<0\)) | <0.05 | 정상. Classing 우수 | 유지 |
| WoE와 부호 반대 | ≥0.05 | 유의성 부족 | 인접 Bin 합병 후 재검정 |
| WoE와 부호 동일 (\(\hat{\beta}>0 \leftrightarrow \text{WoE}>0\)) | 어떤 값이든 | Classing 오류 | Classing 전면 재검토. 해당 변수 투입 금지 |
| WoE와 부호 반대이나 크기 차이 큼 | <0.05 | 과적합 또는 Data Leakage 의심 | 샘플 안정성 재확인. 성과 기간·관찰 시점 재검토 |
