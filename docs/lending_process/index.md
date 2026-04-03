---
title: 여신 프로세스와 신용평가모형의 위치
---

# 여신 프로세스와 신용평가모형의 위치

본 가이드북은 **신용평가모형 개발** — 데이터 준비부터 변수 선정, 모형 학습, 스코어카드 변환, 검증까지 —
을 다루고 있습니다.
그런데 이 모형이 실제 금융기관에서 **어떻게 소비되는지**, 여신(대출) 프로세스 전체에서
어떤 위치를 차지하는지를 조감하면 모형 개발의 맥락을 더 명확하게 이해할 수 있습니다.

!!! warning "일반론적 관점"
    아래 내용은 여신 프로세스의 **일반적인 프레임워크**를 정리한 것입니다.
    실제 프로세스는 금융기관마다 다르며, 저는 모형 개발 영역 외
    심사·사후관리·리스크관리 현업 경험이 없음을 밝힙니다.
    이 챕터의 내용은 Claude Code와 공부하면서 정리한 것으로, 실무와 다를 수 있습니다.

---

## 여신 라이프사이클 전체 그림

금융기관의 대출은 크게 다섯 단계를 거칩니다.

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
          · Score 산출<br>
          · 전략모형
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
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(139,149,161,0.25); border-top:none;">
          · Behavioral Score<br>
          · 한도 증감<br>
          · 조기경보(EWS)<br>
          · Cross/Up-sell<br>
          · 금리 재조정
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; width:20px; min-width:20px; color:#ccc; font-size:0.9rem;">→</div>

      <!-- ④ 부실·추심 -->
      <div style="flex:1; display:flex; flex-direction:column;">
        <div style="text-align:center; padding:8px 6px; border-radius:8px 8px 0 0; font-size:0.72rem; font-weight:600; background:rgba(245,159,0,0.10); color:#f59f00; border:1px solid rgba(245,159,0,0.25); border-bottom:none;">
          <span style="font-size:0.82rem; font-weight:700; display:block; margin-bottom:1px;">④</span>
          부실·추심
        </div>
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(139,149,161,0.25); border-top:none;">
          · 연체 관리<br>
          · Collection Score<br>
          · 회수 전략<br>
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
        <div style="flex:1; padding:10px 8px; border-radius:0 0 8px 8px; font-size:0.7rem; line-height:1.7; background:rgba(245,159,0,0.06); border:1px solid rgba(139,149,161,0.25); border-top:none;">
          · IFRS 9 ECL<br>
          · Basel 자기자본<br>
          · 스트레스 테스트<br>
          · 모형 검증<br>
          · 감독당국 보고
        </div>
      </div>

    </div>
  </div>
</div>

각 단계에서 신용평가모형의 산출물(p̂, Score, 등급)이 어떻게 활용되는지 하위 페이지에서 살펴봅니다.

---

## p̂ / PD 활용 방식 종합

아래 표에서 **서열**과 **절대값**은 다음을 의미합니다.

- **서열** — p̂을 변환한 **평점(Score)과 등급**. 누가 더 위험한지의 **순서**만 중요하며, p̂이 정확히 몇 %인지는 관여하지 않음.
- **절대값** — p̂ = 0.03이냐 0.05이냐처럼, **확률값 자체**가 금액 산출(충당금, 자본 등)에 직접 투입됨.

| 영역 | PD 사용 방식 | 서열/절대값 | calibration 민감도 |
|---|---|:---:|:---:|
| 승인/거절 | cross-matrix 기반 | 서열 | 낮음 |
| 금리 산정 | EL 계산에 PD 투입 | 절대값 | **높음** |
| 한도 산정 | 등급별 한도 밴드 | 서열 | 낮음~중간 |
| Behavioral Score | 리스크 재평가 | 서열 | 낮음 |
| 조기경보 (EWS) | 임계값 기반 플래그 | 서열 | 중간 |
| Collection Score | 회수 가능성 예측 | 서열 | 낮음 |
| IFRS 9 ECL | PD → 충당금 산출 | **절대값** | **높음** |
| Basel RWA | PD → 자본 산출 | **절대값** | **높음** |
| 스트레스 테스트 | PD 변동폭 추정 | **절대값** | **높음** |
| 모형 정기 검증 | 예측 PD vs 실현율 | **절대값** | **높음** |

!!! tip "핵심 구분"
    **서열만 필요한 단계** — 승인/거절, Behavioral Score, Collection Score, 한도 밴드
    → ML vs Logistic의 p̂ 분포 차이가 문제되지 않음

    **절대값이 필요한 단계** — 금리(EL), IFRS 9, Basel, 스트레스 테스트
    → calibration 차이가 실질적 영향을 미침
    → 단, 등급별 PD 테이블을 경유하면 calibration 이슈가 등급화 단계에서 흡수됨
