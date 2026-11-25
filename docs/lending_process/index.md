---
title: 여신 프로세스와 신용평가모형의 위치
---

# 여신 프로세스와 신용평가모형의 위치

본 가이드북은 **신용평가모형 개발** — 데이터 준비부터 변수 선정, 모형 학습, 스코어카드 변환, 검증까지 —
을 다루고 있다.
그런데 이 모형이 실제 금융기관에서 **어떻게 소비되는지**, 여신(대출) 프로세스 전체에서
어떤 위치를 차지하는지를 조감하면 모형 개발의 맥락을 더 명확하게 이해할 수 있다.

!!! warning "일반론적 관점"
    아래 내용은 여신 프로세스의 **일반적인 프레임워크**를 정리한 것이다.
    실제 프로세스는 금융기관마다 다르며,
    심사(여신심사부), 사후관리(채권관리부·추심부), 리스크관리(리스크관리부) 등
    각 단계는 해당 현업 부서가 깊은 전문성을 가지고 있는 영역이다.
    이 챕터는 모형 개발자의 관점에서 공개 자료를 바탕으로 정리한 것이므로,
    기관별 실무와 다를 수 있다.

---

## 여신 라이프사이클 전체 그림

금융기관의 대출은 크게 다섯 단계를 거친다.

<div style="margin: 28px 0 12px;">
  <!-- 범례 -->
  <div style="display:flex; gap:16px; margin-bottom:18px; font-size:0.7rem; flex-wrap:wrap;">
    <span style="display:flex; align-items:center; gap:6px;">
      <span style="width:11px;height:11px;border-radius:3px;background:#1bb76e;display:inline-block;"></span>
      가이드북에서 다룬 영역
    </span>
    <span style="display:flex; align-items:center; gap:6px;">
      <span style="width:11px;height:11px;border-radius:3px;background:#f59f00;display:inline-block;"></span>
      가이드북 범위 밖 (일반론)
    </span>
    <span style="display:flex; align-items:center; gap:6px;">
      <span style="font-weight:700; color:#1bb76e;">초록 굵은 글씨</span>
      범위 밖 단계 중 가이드북 본문에서 다루는 항목
    </span>
  </div>

  <!-- 플로우 다이어그램 — 모바일에서 가로 스크롤 -->
  <div style="overflow-x:auto; padding-bottom:8px;">
    <div style="display:flex; align-items:stretch; gap:0; min-width:750px;">

      <!-- ① 모형 개발 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(27,183,110,0.10); color:#1bb76e; border:1px solid rgba(27,183,110,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">①</span>
          모형 개발
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(27,183,110,0.06); border:1px solid rgba(27,183,110,0.25); border-top:none;">
          · 데이터 수집·가공<br>
          · 변수 선택<br>
          · LR / ML 학습<br>
          · 성능 검증<br>
          · p̂→Score→등급<br>
          · 등급별 PD 산정
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; width:20px; min-width:20px; color:#ccc; font-size:0.9rem;">→</div>

      <!-- ② 심사·승인 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(245,159,0,0.10); color:#f59f00; border:1px solid rgba(245,159,0,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">②</span>
          심사·승인
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(245,159,0,0.25); border-top:none;">
          · 승인/거절<br>
          · 금리 산정<br>
          · 한도 산정
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; width:20px; min-width:20px; color:#ccc; font-size:0.9rem;">→</div>

      <!-- ③ 사후 관리 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(245,159,0,0.10); color:#f59f00; border:1px solid rgba(245,159,0,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">③</span>
          사후 관리
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(245,159,0,0.25); border-top:none;">
          · Behavioral Score<br>
          · 한도 증감<br>
          · 조기경보(EWS)<br>
          · Cross/Up-sell
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; width:20px; min-width:20px; color:#ccc; font-size:0.9rem;">→</div>

      <!-- ④ 부실·추심 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(245,159,0,0.10); color:#f59f00; border:1px solid rgba(245,159,0,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">④</span>
          부실·추심
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(245,159,0,0.25); border-top:none;">
          · 연체 관리<br>
          · Collection Score<br>
          · 채무조정<br>
          · 채권 매각·상각
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; width:20px; min-width:20px; color:#ccc; font-size:0.9rem;">→</div>

      <!-- ⑤ 리스크·규제 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(245,159,0,0.10); color:#f59f00; border:1px solid rgba(245,159,0,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">⑤</span>
          리스크·규제
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(245,159,0,0.25); border-top:none;">
          · IFRS 9 ECL<br>
          · Basel 자기자본<br>
          · 스트레스 테스트<br>
          · <span style="font-weight:700; color:#1bb76e;">모형 검증·모니터링</span>
        </div>
      </div>

    </div>
  </div>
</div>

각 단계에서 신용평가모형의 산출물(p̂, Score, 등급, 등급별 PD)이 어떻게 활용되는지 하위 페이지에서 살펴본다.

| 단계 | 페이지 | 모형 산출물과의 접점 |
|---|---|---|
| ① 모형 개발 | [모형 개발](model-development.md) | 가이드북 각 파트와의 매핑, 서열과 캘리브레이션의 역할 분담 |
| ② 심사·승인 | [심사·승인](underwriting.md) | 승인/거절 cross-matrix, 금리의 신용원가(EL), 한도 밴드 |
| ③ 사후 관리 | [사후 관리](portfolio-mgmt.md) | 행동평점(BS), 한도 재심사, 조기경보 |
| ④ 부실·추심 | [부실·추심](collection.md) | Collection Score, 채무조정, 매각·상각과 LGD |
| ⑤ 리스크·규제 | [리스크·규제](risk-regulation.md) | IFRS 9 충당금, Basel 자본, 정기 검증 |
| 용어 | [PD](terms/pd.md) · [LGD](terms/lgd.md) · [EAD](terms/ead.md) | 신용원가(EL)의 세 축과 PD의 층위 |

---

## p̂ / PD 활용 방식 종합

아래 표에서 **서열**과 **절대값**은 다음을 의미한다.

- **서열** — p̂을 변환한 **평점(Score)과 등급**. 누가 더 위험한지의 **순서**만 중요하며, p̂이 정확히 몇 %인지는 관여하지 않는다.
- **절대값** — p̂ = 0.03이냐 0.05이냐처럼, **확률값 자체**가 금액 산출(충당금, 자본 등)에 직접 투입된다.

| 영역 | PD 사용 방식 | 서열/절대값 | 캘리브레이션 민감도 |
|---|---|:---:|:---:|
| 승인/거절 | cross-matrix 기반 | 서열 | 낮음 (손익 기반 Cut-off를 쓰면 중간) |
| 금리 산정 | EL 계산에 PD 투입 | 절대값 | **높음** |
| 한도 산정 | 등급별 한도 밴드 | 서열 | 낮음~중간 |
| Behavioral Score | 리스크 재평가 | 서열 | 낮음 |
| 조기경보 (EWS) | 임계값 기반 플래그 | 서열 | 중간 |
| Collection Score | 회수 가능성 예측 | 서열 | 낮음 |
| IFRS 9 ECL | PD → 충당금 산출 | **절대값** | **높음** |
| Basel RWA | PD → 자본 산출 | **절대값** | **높음** |
| 스트레스 테스트 | PD 변동폭 추정 | **절대값** | **높음** |
| 모형 정기 검증 | 예측 PD vs 실현 부도율 | **절대값** | **높음** |

!!! tip "핵심 구분 — 서열은 모형, 수준은 캘리브레이션"
    **서열만 필요한 단계** — 승인/거절, Behavioral Score, Collection Score, 한도 밴드.
    이 단계에서는 점수의 순서만 쓰므로, 같은 차주 집단에 대해 서열이 같다면
    로지스틱 회귀와 ML 모형의 p̂ **수준·퍼짐**이 서로 달라도 의사결정에 영향을 주지 않는다.

    **절대값이 필요한 단계** — 금리(EL), IFRS 9, Basel, 스트레스 테스트, 정기 검증.
    여기서는 p̂ 또는 등급에 붙는 PD 숫자가 곧 금액이 되므로 **캘리브레이션**(예측 PD와 실현 부도율의 일치)이 실질적 영향을 미친다.

    실무에서 흔한 방식은 모형의 p̂을 직접 쓰지 않고 **등급별 PD 테이블**을 경유하는 것이다.
    이때 각 등급의 PD를 해당 등급의 **실현 부도율(또는 CT 기준 장기 평균)로 다시 산정**한다면,
    모형 p̂의 수준 차이는 등급화 단계에서 흡수된다. 반대로 등급별 PD를 p̂의 평균으로 붙인다면
    p̂의 캘리브레이션 오차가 그대로 남는다. 등급별 PD를 붙이는 방법은 [PD 용어 정리](terms/pd.md)에서 다룬다.
