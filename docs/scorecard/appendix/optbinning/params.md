# 핵심 옵션의 의미

이 페이지는 optbinning의 주요 옵션이 [최적화 정식화](optimizer.md)의 **어느 제약에 해당하고, 왜 필요하며, 어떻게 정하는가**를 정리한다. 옵션 이름은 공식 문서와 대조할 수 있도록 괄호로만 표기한다. 기본값은 [공식 문서](https://gnpalencia.org/optbinning/binning_binary.html) 기준이다.

## 2.1 단조성 제약 (`monotonic_trend`)

가장 중요한 옵션이다. **Event rate(불량률)** 의 방향을 제약한다. WoE 방향이 아니다.

| 값 | 의미 (Event rate 기준) | 가이드북 WoE 기준 | 사용 예 |
|----|------|------|----------|
| auto (기본값) | 학습된 분류기로 방향을 자동 결정 | — | 방향을 모를 때의 첫 탐색 |
| ascending | Event rate 단조 증가 | WoE 단조 감소 | 사용률↑ → 위험↑, 부채비율↑ → 위험↑ |
| descending | Event rate 단조 감소 | WoE 단조 증가 | 소득↑ → 위험↓, 업력↑ → 위험↓ |
| peak | 중간 구간에서 Event rate 최대(역U형) | 중간에서 WoE 최소 | 드문 경우. 업무 논리 확인 필수 |
| valley | 중간 구간에서 Event rate 최소(U형) | 중간에서 WoE 최대 | 재고자산회전율처럼 너무 낮아도 높아도 위험한 비율 |
| concave / convex | Event rate 곡선의 오목·볼록 제약 | — | 곡률까지 통제하고 싶을 때 |
| auto_asc_desc | ascending·descending 중 IV가 큰 쪽 자동 선택 | — | U형·역U형을 배제하고 싶을 때 |
| *_heuristic | peak/valley/auto의 변곡점을 휴리스틱으로 결정 | — | prebin이 많을 때(속도) |

!!! tip "auto는 '탐색용'으로만"
    공식 문서는 auto를 "머신러닝 분류기를 이용해 IV를 최대화하는 추세를 자동으로 결정"한다고 설명한다. 편리하지만, 신용평가에서는 방향이 **업무 논리로 먼저 정해져야 하는** 경우가 대부분이다. auto의 결과는 "데이터가 말하는 방향"을 확인하는 참고로만 쓰고, 최종 모형에는 ascending/descending을 명시하는 것을 권한다. 데이터가 고른 방향이 업무 논리와 반대라면 그 자체가 표본·정의 오류의 신호다.

!!! warning "흔한 실수 — 방향을 WoE로 착각"
    "소득이 높을수록 WoE가 커지니 ascending"이라고 설정하면, 실제로는 **소득이 높을수록 불량률이 커지도록** 강제된다. 결과 Binning Table의 불량률 열을 보고 방향을 반드시 재확인한다.

---

## 2.2 표본 크기 제약

| 옵션 | 의미 | 기본값 |
|---|---|---|
| 최소 Bin 비중 (`min_bin_size`) | 각 Bin이 전체 표본에서 차지해야 하는 최소 비율 | 미지정 시 Fine Bin 최소 비중(5%)과 같음 |
| 최대 Bin 비중 (`max_bin_size`) | 한 Bin에 표본이 과도하게 몰리는 것 방지 | 제한 없음 |
| 최소 Bad / Good 건수 (`min_bin_n_event`, `min_bin_n_nonevent`) | 각 Bin의 최소 불량·정상 건수 | 제한 없음 |

**왜 필요한가.** WoE는 \(\ln(\%\text{Good}/\%\text{Bad})\)이므로 Bad 건수가 적은 Bin에서는 Bad 1~2건 차이로 WoE가 크게 흔들린다. 예를 들어 Bad 5건인 Bin은 Bad가 4건이 되면 WoE가 \(\ln(5/4) \approx 0.22\)만큼 움직인다. 그런 Bin의 WoE는 OOT에서 재현되지 않는다. 비율 기준(최소 비중)만으로는 저불량 포트폴리오에서 Bad 건수가 부족할 수 있으므로, **최소 Bad 건수를 함께** 거는 것이 안전하다. Coarse Bin 기준값(각 Bin ≥ 5%, Bad 건수 최소 기준 병기)은 본문 [구간화 결정 기준](../../part3_variable_selection/classing/decision-criteria.md)을 따른다.

---

## 2.3 Bin 수 제약 (`min_n_bins`, `max_n_bins`)

최대 Bin 수를 지정하지 않으면 공식 문서상 Bin 수는 "0과 최대 Fine Bin 수 사이의 값"이 된다. 여기서 주의할 점은 **IV는 구간을 합칠수록 줄어들기만 한다**는 사실이다([최적화 엔진 1.3절](optimizer.md) 예시의 1위 참조). 따라서 Bin 수 상한이 없으면 Solver는 다른 제약(단조성·최소 표본·p-value)이 허용하는 한 **가능한 많은 Bin**을 남긴다. "Solver가 적당한 Bin 수를 알아서 고른다"가 아니라, **Bin 수는 다른 제약이 결정한다**.

실무에서는 해석 가능성과 안정성을 위해 상한을 명시한다. 스코어카드에서는 변수당 5개 안팎이 흔하다([기준값 정본](../../part3_variable_selection/classing/decision-criteria.md)). 해석 가능한 ML의 조건과 운영 평점표의 구간 수 관행은 [해석 가능성 페이지](../../../ml/part4_interpretation/interpretability.md)를 참고한다.

---

## 2.4 유의성 제약 (`max_pvalue`, `max_pvalue_policy`)

| 옵션 | 의미 |
|---------|------|
| 최대 p-value (`max_pvalue`) | 이웃한 두 Bin의 불량률 차이에 대한 Z-test p-value 상한. 기준을 넘는 쌍은 동시에 채택될 수 없다(최적화 제약). 기본값은 미적용 |
| 비교 범위 (`max_pvalue_policy`) | consecutive(기본값): 바로 이웃한 쌍만 / all: 모든 Bin 쌍 |

**왜 필요한가.** 불량률 차이가 통계적으로 구분되지 않는 두 Bin은 서로 다른 점수를 줄 근거가 약하다. 이 제약은 본문 [비유의 Bin 처리 기준](../../part3_variable_selection/univariate_lr/significance-test.md)("인접 Bin 간 차이가 비유의면 합병 검토")을 최적화 단계에서 미리 강제하는 장치다.

!!! warning "최적화 내부 검정과 사후 검정은 다르다"
    최적화 내부에서는 Event rate 차이에 대한 **Z-test**를 쓴다(연속형 타겟은 T-test). 결과표의 사후 분석에서는 카이제곱 또는 Fisher 정확 검정을 고를 수 있다. 같은 Bin 쌍이라도 검정 방식에 따라 p-value가 조금씩 다르므로, 보고서에는 어떤 검정을 썼는지 명시한다. 대표본에서는 거의 모든 차이가 유의해지므로 p-value 제약은 사실상 작동하지 않는 경우가 많다. 이때는 최소 Bin 비중·Bin 수 상한이 실질적인 제약이 된다.

---

## 2.5 Special Codes와 Missing 처리

- **Special Codes**(`special_codes`): 지정한 값(예: −9, −99 같은 센티널 값)은 최적화 대상에서 빠지고 **별도의 Special Bin**이 된다. 여러 코드를 묶어 그룹별로 별도 Bin을 만들 수도 있다.
- **Missing**(NaN): 자동으로 별도 Bin이 된다.
- 두 Bin 모두 단조성 제약의 대상이 아니다. 즉 Special/Missing Bin의 불량률은 정상 구간의 추세와 무관하게 나온다.

!!! danger "변환 시 기본값 0 = '평균 위험' 처리 — 실무 함정"
    WoE 변환에서 Special·Missing의 기본값은 **0**이다(공식 문서: 변환 옵션 기본값 0). WoE = 0은 "모집단 평균과 같은 위험"을 뜻한다. 그런데 실무에서 특수코드·결측은 **그 자체가 위험 신호**인 경우가 많다(예: 소득 미확인, 신용이력 없음). 이런 구간에 0을 주면 위험한 차주가 평균 점수를 받는다.

    - 표본이 충분하면 변환 옵션을 **empirical**(해당 Bin의 실제 WoE 사용)로 두는 것을 검토한다.
    - 표본이 부족하면 업무 판단으로 보수적인 값(예: 가장 위험한 정상 구간의 WoE)을 부여하고 근거를 문서화한다.
    - Special Value의 WoE 처리 방침은 본문 [Fine Classing](../../part3_variable_selection/classing/fine-classing.md)의 특수코드 처리와 일관되게 정한다.

---

## 2.6 범주형 변수 처리 (`dtype`, `cat_cutoff`)

범주형 변수로 지정하면 라이브러리는 각 범주의 Event rate를 계산해 **범주를 Event rate 순으로 정렬한 뒤** 연속형처럼 인접 범주끼리 병합한다. 그래서 업종 코드처럼 순서가 없는 변수도 "위험도 순서"를 기준으로 묶인다.

희귀 범주 기준(`cat_cutoff`)을 지정하면, 출현 비율이 기준 미만인 범주들은 최적화 전에 하나의 **"others"** 그룹으로 묶인다(기본값은 미적용). 희귀 범주는 Event rate 추정이 불안정해 정렬 순서 자체가 우연에 좌우되기 때문이다. 단, others 그룹은 성격이 다른 범주들이 섞인 집합이므로 다음을 점검한다.

- others의 표본이 커서 결과에 큰 영향을 주는지
- 신규 범주(학습 때 없던 업종 코드)가 운영 중 들어오면 어느 Bin으로 갈지 — 보통 others 또는 별도 규칙으로 처리하고 문서화한다

---

## 2.7 정규화 (`gamma`)

공식 문서는 이 옵션을 "dominating bins의 수를 줄이는 정규화 강도"로 설명한다(기본값 0). 여기서 dominating bin은 **표본이 한 Bin에 과도하게 쏠린 경우**를 말한다. 값이 클수록 목적함수에서 Bin 크기 불균형에 대한 페널티가 커져, 표본이 여러 Bin에 더 고르게 나뉘도록 유도한다.

- 언제 쓰나: 최적해가 "한 Bin에 표본 60%, 나머지는 잘게 쪼갠" 형태로 나와 운영상 변별이 약해 보일 때
- 주의: 페널티만큼 IV를 희생한다. 값은 몇 단계로 바꿔 보며 IV 감소폭과 Bin 분포 변화를 함께 보고 정한다. 최대 Bin 비중 제약으로 같은 목적을 더 직접적으로 달성할 수도 있다.
