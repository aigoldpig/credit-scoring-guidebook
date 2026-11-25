# 기호 용어집

본 가이드북 전체에서 사용하는 주요 기호와 약어를 정리한다. 부호 규약은 [부록 첫 페이지](index.md)의 규약 박스를 따른다.

## 확률·오즈 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| \(p\) 또는 \(P(y=1 \mid \mathbf{x})\) | 불량(Bad) 확률, y=1=Bad | 전체 |
| \(\text{Odds} = p/(1-p)\) | Bad Odds — 정상 대비 불량의 비율(불량/정상) | Part 1~4 |
| \(\text{Good Odds} = (1-p)/p\) | Good Odds — 불량 대비 정상의 비율(정상/불량) | Part 5 스코어카드 변환 |
| \(\text{Logit}(p) = \ln(p/(1-p))\) | 로그 오즈(Log-Odds). 모형 적합은 항상 이 Bad logit 기준 | 이론, 모델링 |
| \(\eta = \beta_0 + \boldsymbol{\beta}^\top \mathbf{x}\) | 선형 예측자 (Linear Predictor) | 이론, MLE |
| \(\mathcal{L}(\boldsymbol{\beta})\) | 우도함수 (Likelihood Function) | 이론 |
| \(\ell(\boldsymbol{\beta}) = \ln \mathcal{L}\) | 로그우도함수 (Log-Likelihood). 부록 B에서는 \(\log L\)로도 표기 | 이론, MLE, 부록 B |

## 모형 계수 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| \(\beta_0\) | 절편 (Intercept) | 모델링 |
| \(\beta_j\) | 변수 \(j\)의 회귀계수. WoE 투입 시 정상 부호는 음수(\(\beta_j<0\)) | 전체 |
| \(\hat{\beta}\) | MLE로 추정된 계수 | 이론, 단변량 LR |
| \(\gamma_{jb}\) | 더미 방식에서 변수 \(j\) 구간 \(b\)의 계수(기준 구간 대비 로짓 차이) | 부록 B |
| \(\text{SE}(\hat{\beta}_j)\) | 계수의 표준오차 | 유의성 검정 |
| \(e^{\beta_j}\) | 오즈비 (Odds Ratio) | 이론 |
| \(I(\boldsymbol{\beta}) = \mathbf{X}^\top\mathbf{W}\mathbf{X}\) | Fisher 정보행렬 | 부록 B |

## 검정 통계량 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| \(z = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)\) | Wald 검정 통계량 | 단변량 LR, 유의성 검정 |
| \(U_j\) | Score(점수) — 로그우도의 \(\beta_j\)에 대한 1차 미분 | 부록 B |
| \(\chi^2\) | 카이제곱 통계량 | 유의성 검정 |
| \(p\text{-value}\) | 유의확률 — 귀무가설 하에서 관측값 이상의 극단적 결과가 나올 확률 | 유의성 검정 |
| SLENTRY / SLSTAY | Stepwise 진입·잔류 유의수준 | 부록 B |
| AIC / BIC | 정보량 기준 \(-2\log L + 2k\) / \(-2\log L + k\ln n\) (\(k\)=절편 포함 모수 수) | 부록 B |

## WoE·IV 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| \(n_{G,i}\), \(n_{B,i}\) | Bin \(i\)의 Good/Bad 건수 | WoE/IV |
| \(N_G\), \(N_B\) | 전체 Good/Bad 건수 | WoE/IV |
| \(\%\text{Good}_i = n_{G,i}/N_G\) | 전체 Good 중 Bin \(i\) 비중 | WoE/IV |
| \(\%\text{Bad}_i = n_{B,i}/N_B\) | 전체 Bad 중 Bin \(i\) 비중 | WoE/IV |
| \(\text{WoE}_i = \ln(\%\text{Good}_i / \%\text{Bad}_i)\) | Weight of Evidence (우량 구간 > 0) | WoE/IV |
| \(\text{IV} = \sum (\%\text{Good}_i - \%\text{Bad}_i) \times \text{WoE}_i\) | Information Value | WoE/IV |

## 스코어카드 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| Anchor | Anchor Score — 기준 Odds \(\theta_0\)에서의 목표 점수 | 스코어카드 변환 |
| Offset | Score 수식의 상수항. \(\text{Offset} = \text{Anchor} - \text{Factor}\cdot\ln\theta_0\) | 스코어카드 변환 |
| Factor \(= \text{PDO}/\ln 2\) | 스케일링 상수. 부록 일부에서는 \(B\)로 표기 | 스코어카드 변환 |
| \(\theta_0\) | 기준 Odds (Good:Bad) | 스코어카드 변환 |
| PDO | Points to Double the Odds | 스코어카드 변환 |

## LOWESS 관련 (부록 C)

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| \(T(x)\) | 재무비율 \(x\)의 단변량 부도확률(변환값) | 부록 C |
| \(f\) | LOWESS 대역폭(span) — 국소 회귀에 쓰는 이웃의 비율 | 부록 C |
| \(q = \text{round}(f \cdot n)\) | 국소 회귀에 쓰는 이웃 수 | 부록 C |
| \(h_i\) | 평가점 \(x_i\)에서 \(q\)번째로 가까운 이웃까지의 거리 | 부록 C |
| \(W(u) = (1-\lvert u\rvert^3)^3\) | Tricube 거리 가중함수 (\(\lvert u\rvert<1\)) | 부록 C |
| \(\Phi\) | 표준정규 누적분포함수(프로빗) | 부록 C |

## 성능 지표 관련

| 기호 | 정의 | 사용 위치 |
|------|------|----------|
| KS | Kolmogorov-Smirnov 통계량 — Good/Bad 누적분포 최대 차이 | 성능 평가 |
| AR (Gini) | Accuracy Ratio — CAP 곡선 기반 누적 변별력, \(\text{AR} = 2 \times \text{AUC} - 1\) | 성능 평가 |
| AUC (AUROC) | Area Under ROC Curve — AR 환산용 보조 지표 | 성능 평가 |
| PSI | Population Stability Index — 점수 분포 안정성 | 성능 평가 |
| ROC | Receiver Operating Characteristic — 민감도 vs 1-특이도 곡선 | 성능 평가 |
| CAP | Cumulative Accuracy Profile — AR 산출의 기초 곡선 | 성능 평가 |
| CSI | Characteristic Stability Index — 개별 변수의 분포 안정성 지표 | 성능 평가 |
| VIF | Variance Inflation Factor — 다중공선성 진단 | 모델링 |
| Concordance | 일치쌍 — 모형이 Bad를 더 높은 확률로 예측한 Good-Bad 쌍의 비율 | 성능 평가 |
| Discordance | 불일치쌍 — Concordance의 반대 | 성능 평가 |

## 약어

| 약어 | 풀이 | 비고 |
|------|------|------|
| CSS | Credit Scoring System | 신용평가모형 |
| AS | Application Scoring | 신청 시점 평가 |
| BS | Behavioral Scoring | 기존 고객 행동 평가 |
| CB | Credit Bureau | 개인신용평가회사(CB사, NICE평가정보·KCB 등). 한국신용정보원(종합신용정보집중기관)과 다르다 |
| IRB | Internal Ratings-Based approach | Basel 내부등급법 |
| MLE | Maximum Likelihood Estimation | 최대우도추정 |
| LRT | Likelihood Ratio Test | 우도비 검정 |
| OOT | Out-of-Time | 시간 외 검증 샘플 |
| PD | Probability of Default | 부도확률 |
| TTC | Through-the-Cycle | 경기순환 조정 |
| EDF | Expected Default Frequency | 기대부도빈도 (Moody's 등에서 사용) |
| LGD | Loss Given Default | 부도시 손실률 |
| EAD | Exposure At Default | 부도시 익스포저 |
| LOWESS | Locally Weighted Scatterplot Smoothing | 국소 가중 산점도 평활법 |
| MoC | Margin of Conservatism | 보수성 가산 |

## 주요 용어

| 용어 | 정의 | 사용 위치 |
|------|------|----------|
| Fine Classing | 연속변수를 다수의 세분화 구간으로 초기 분할하는 단계 | 변수 선정 |
| Coarse Classing | Fine Classing 결과를 유사 구간끼리 병합하여 최종 구간을 확정하는 단계 | 변수 선정 |
| 단조성 (Monotonicity) | WoE 또는 Bad Rate가 구간 순서대로 일관되게 증가 또는 감소하는 성질 | 변수 선정 |
| Observation Window | 변수 산출에 사용하는 과거 데이터 기간 | 개요 |
| Performance Window | 목표변수(Good/Bad) 판정을 위한 관찰 기간 | 개요 |
| Vintage 분석 | 동일 시점 대출 코호트의 시간 경과별 부도율을 추적하는 분석 기법 | 개요 |
| Roll Rate | 연체 단계 간 전이율 (예: 정상→30일 연체, 30일→60일 연체) | 개요 |
| Target Leakage | 예측 시점에 알 수 없는 미래 정보가 학습 데이터에 섞여 성능이 낙관적으로 부풀려지는 오류 | 개요 |
| 미니모델링 | 현업 용어. LOWESS 기반으로 재무비율별 단변량 부도확률을 추정·변환하는 기법 | 부록 C |
| 기불량 | 관찰 시점에 이미 Bad 정의 수준의 상태(예: 90일 이상 미해제 연체)인 차주. 개발 표본에서 제외한다 | 개요(요건 정의) |
| 등급 상한 (Grade Cap) | 정해진 사유가 있으면 모형 점수와 무관하게 받을 수 있는 최고 등급을 제한하는 자동 규칙. 항목별로 상한을 정하고 여러 항목에 걸리면 가장 낮은 상한을 적용한다 | 스코어카드 변환 & 등급화 |
