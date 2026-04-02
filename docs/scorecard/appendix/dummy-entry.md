# 더미 변수 방식

## 3.1 수리적 구조

변수 \(j\)가 \(m_j\)개의 구간을 가질 때, 구간 0을 기준(Reference)으로 두고 나머지 \((m_j-1)\)개 구간에 대한 더미를 생성한다.

$$
\text{logit}(p_i) = \beta_0 + \sum_{j=1}^{k} \sum_{b=1}^{m_j-1} \gamma_{jb} \cdot \mathbf{1}[\text{변수}_j = b] \tag{A.4}
$$

\(\gamma_{jb}\)는 "변수 \(j\)가 기준 구간(0번) 대비 \(b\)번 구간에 속할 때의 로짓 차이"다.

??? note "SAS 구현 코드"

    === "Stepwise 변수 선택"

        ```sas
        PROC LOGISTIC DATA=적합데이터;
          WHERE val_gb=0 AND target IN (0, 1);
          CLASS &VAR_LIST. / PARAM=REFERENCE REF=FIRST;
          MODEL target = &VAR_LIST. / CTABLE PARMLABEL
            SELECTION=STEPWISE SLSTAY=0.1;
        RUN;
        ```

    === "최종 모형 적합"

        ```sas
        PROC LOGISTIC DATA=적합데이터 OUTMODEL=LR_FIT;
          WHERE val_gb=0 AND target IN (0, 1);
          CLASS &VAR_LIST. / PARAM=REFERENCE REF=FIRST;
          MODEL target = &VAR_LIST. / CTABLE PARMLABEL;
        RUN;
        ```

    `PARAM=REFERENCE REF=FIRST`는 숫자 오름차순 정렬 후 가장 작은 값(0번)을 기준 범주로 지정한다.

---

## 3.2 기준점과 계수 해석

구간 번호가 {0, 1, 2}이면 가장 작은 값인 0번(DSR≥70%, 최위험 구간)을 기준 범주(Reference)로 삼는다.

!!! info "기준점의 해석 장점"
    기준 구간(0번, 최위험)의 효과는 절편 \(\beta_0\)에 흡수된다. 나머지 \(\gamma_{j1}\), \(\gamma_{j2}\)는 "최위험 대비 얼마나 개선되었는가"를 직접 나타낸다.

    DSR이 안전한 구간(2번)일수록 \(\gamma_{j2} > \gamma_{j1} > 0\)이 되어야 한다. 이 부호 방향이 역전되면 구간화 재검토 신호다.

    감독기관 보고 시 "최위험 구간 대비 각 구간의 로짓 개선량"으로 직관적 설명이 가능하다.

---

## 3.3 스코어카드 점수 변환

더미 방식에서 각 구간의 \(\gamma_{jb}\) 자체가 상대적 효과이므로 점수 변환이 직접적이다.

$$
\text{부분점수}_{j,0} = 0 \quad (\text{기준 구간}) \tag{A.5}
$$

$$
\text{부분점수}_{j,b} = -B \cdot \gamma_{jb} \quad (b \geq 1) \tag{A.6}
$$

!!! warning "부호 주의"
    기준이 최위험 구간(0번)이고 안전 구간의 \(\gamma\)는 양수이므로, \(-B \times \gamma\)를 적용하면 안전 구간의 부분점수는 **음수**가 된다. 스코어카드 총점 설계 시 기준 구간 절대점수를 먼저 고정한 후, 나머지 구간을 상대점수로 가산하는 방식으로 처리한다.
