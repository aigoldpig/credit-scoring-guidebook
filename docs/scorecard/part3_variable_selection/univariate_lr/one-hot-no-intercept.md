# One-Hot Encoding + No Intercept 회귀

## 2.1 구간 변수를 회귀에 넣는 세 가지 방법

범주형 변수(또는 구간으로 나뉜 연속형 변수)를 로지스틱 회귀에 투입하는 방법은 세 가지다. 셋은 **같은 적합도**를 주지만 계수의 의미가 다르다.

| 구분 | ① 일반 더미 (k−1 + 절편) | ② One-Hot (k개 + No Intercept) | ③ WoE 치환 (+ 절편) |
|------|----------------------|-------------------------------------|------------------|
| **설명변수** | \(k-1\)개 더미 (기준 구간 1개 제외) | \(k\)개 더미 (모든 구간) | WoE 값 1개 |
| **절편** | 포함 | 제거 | 포함 |
| **Full-rank 여부** | \(k-1\)개 + 절편 → Full-rank | \(k\)개, 절편 없음 → Full-rank | Full-rank |
| **계수의 의미** | 기준 구간 대비 Bad log-odds **차이** \(\gamma_i\) | 구간 자체의 Bad log-odds \(\hat\beta_i\) | WoE 1단위당 Bad log-odds 변화 |
| **WoE와의 관계** | \(\gamma_i = -(\text{WoE}_i - \text{WoE}_{\text{기준}})\) | \(\hat{\beta}_i = -\text{WoE}_i + \ln(N_B/N_G)\) | \(\hat\beta = -1\), \(\hat\beta_0 = \ln(N_B/N_G)\) |
| **주 용도** | 다변량 더미 모형 | 단변량 구간별 검정 | 스코어카드 본 모형 |

세 관계식은 모두 근사가 아니라 **정확히** 성립한다([증명](beta-woe-proof.md)).

!!! tip "일반 더미처리(k−1)에서 기준 구간을 빼는 이유"
    \(k\)개 구간에 \(k\)개의 더미를 만들면 **모든 고객이 정확히 하나의 더미에만 해당**하므로 \(D_1 + D_2 + \cdots + D_k = 1\)이 항상 성립한다. 이는 절편 열(모든 원소가 1인 열)과 **완전한 선형 종속**이어서 설계 행렬(Design Matrix)이 Full-rank가 아니게 되고, 계수를 유일하게 추정할 수 없다(완전 다중공선성). 그래서 한 구간을 빼고 \(k-1\)개만 넣는다. 빠진 구간이 기준(Baseline)이 되고, 나머지 계수 \(\gamma_i\)는 기준 대비 Bad log-odds 차이가 된다.

!!! tip "k−1 방식의 계수 부호"
    기준 구간을 **가장 위험한 구간**(예: Bin 1)으로 두면, 나머지 구간은 기준보다 안전하므로 \(\gamma_i\)는 **음수**가 정상이다(Bad log-odds가 기준보다 낮음). \(\gamma_i\)가 양수면 "기준보다 더 위험한 구간"이라는 뜻이다. k−1 방식은 "각 구간 자체의 위험 수준"을 바로 보여주지 않으므로, 단변량 단계에서 구간별 위험을 한눈에 보기에는 ②가 편하다.

반면 **One-Hot Encoding + No Intercept** 방식은 절편을 없애 다중공선성 문제를 피하면서 \(k\)개 더미를 모두 쓴다. 이때 각 \(\hat{\beta}_i\)는 해당 구간의 Bad log-odds를 직접 추정하므로, 구간별 위험 수준과 그 표준오차를 바로 읽을 수 있다.

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
    ①·②·③은 모두 구간마다 서로 다른 위험 수준을 허용하는 **포화(saturated) 모형**과 같은 적합값을 내므로, 로그우도(그리고 이를 쓰는 LRT)가 **수학적으로 동일**하다. ③은 모수가 2개뿐인데도 포화 모형과 같은 적합을 내는데, 이는 WoE 자체가 같은 데이터의 Bin별 log-odds로 만들어졌기 때문이다. 이 점이 아래 LRT 자유도 문제로 이어진다.

## 2.2 No Intercept 모형의 식과 해

One-Hot 더미를 설명변수로, 절편 없이 로지스틱 회귀를 수행한다(y=1=Bad).

$$
\ln\!\left(\frac{p_j}{1-p_j}\right) = \beta_1 D_{j1} + \beta_2 D_{j2} + \beta_3 D_{j3} + \beta_4 D_{j4} + \beta_5 D_{j5} \tag{1}
$$

고객 \(j\)는 하나의 더미에만 1이므로, 구간 \(i\)에 속한 모든 고객의 예측 log-odds는 \(\beta_i\) 하나로 같다. 최대우도추정의 해는 **구간의 실제 Bad log-odds**이다.

$$
\hat\beta_i = \ln\!\left(\frac{n_{B,i}}{n_{G,i}}\right), \qquad \widehat{SE}(\hat\beta_i) = \sqrt{\frac{1}{n_{B,i}} + \frac{1}{n_{G,i}}} \tag{2}
$$

반복 계산이 필요 없는 닫힌 해이며, 유도는 [다음 절](beta-woe-proof.md)에서 한다.

## 2.3 적합 결과 예시 (매출액 5구간, 원 표본 Good 1,000 / Bad 200)

[WoE 1.3절](../woe_iv/woe.md)의 데이터를 세 방식으로 적합한 결과다. 모든 값은 식 (2)와 원 건수로 계산했다.

| Bin | WoE | ② \(\hat\beta_i\) (One-Hot) | SE | ① \(\gamma_i\) (k−1, Bin1 기준) |
|-----|-----|---------------------------|------|-------------------------|
| 1 (1억 미만) | −1.792 | +0.182 | 0.191 | (기준, 절편 = +0.182) |
| 2 (1억~5억) | −0.511 | −1.099 | 0.163 | −1.281 |
| 3 (5억~10억) | 0.000 | −1.609 | 0.173 | −1.792 |
| 4 (10억~50억) | +0.847 | −2.457 | 0.190 | −2.639 |
| 5 (50억 초과) | +0.916 | −2.526 | 0.232 | −2.708 |

- ③ WoE 치환 모형: \(\hat\beta = -1.000\), \(\hat\beta_0 = \ln(200/1000) = -1.609\)
- 세 모형의 로그우도는 모두 −472.64, 절편만 있는 모형은 −540.67 → LRT = 136.07

!!! note "결과 해석"
    - ②의 \(\hat\beta_i + \text{WoE}_i\)는 모든 구간에서 −1.609로 같다. **계산이 맞는지 확인하는 용도**다.
    - ①의 \(\gamma_i\)는 모두 음수다. 기준(최위험 구간)보다 안전하다는 정상 부호다.
    - ③의 \(\hat\beta = -1\)은 Classing이 좋든 나쁘든 개발 표본에서는 항상 나온다. 진단 정보가 아니다.

!!! warning "WoE 치환 모형의 LRT 자유도"
    ③은 추정 모수가 하나(기울기)라서 자유도 1로 LRT를 계산하기 쉽다. 그러나 WoE 값 자체가 같은 표본의 Bin별 log-odds(\(k-1\)개 자유 모수)로 만들어졌으므로, 올바른 자유도는 **\(k-1\)** 이다. 자유도 1로 검정하면 유의성이 과장된다. 변수 유의성은 ①·②와 같은 LRT(자유도 \(k-1\))로 판단한다([유의성 검정](significance-test.md)).

!!! info "균형 표본(Balanced Sampling)이면?"
    개발 표본을 Bad:Good = 1:1로 맞추면 \(\ln(N_B/N_G) = 0\)이 되어 \(\hat\beta_i = -\text{WoE}_i\)가 그대로 성립한다. 위 데이터에서 Bad만 5배 추출(Bad 1,000)했다고 하면 \(\hat\beta\) = (+1.792, +0.511, 0.000, −0.847, −0.916)이다. 이 경우 Bin 불량률은 85.7%, 62.5%, 50.0%, 30.0%, 28.6%로 바뀌므로, 원 표본의 불량률(54.5%, 25.0%, …)과 섞어 쓰면 안 된다. WoE는 Good·Bad 각각의 분포로 계산하므로 표본 비율이 바뀌어도 변하지 않는다.
