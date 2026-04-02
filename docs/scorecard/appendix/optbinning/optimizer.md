# 최적화 엔진 — MIP/CP로 IV 최대화

## 1.1 문제 정식화

Pre-binning으로 생성된 \(n\)개 Fine Bin을 **어떻게 합병해야 IV가 최대**가 되는지를 수학적 최적화로 푼다.

**목적함수:**

$$\max \sum_{i,j} V_{ij} \cdot x_{ij} \tag{1}$$

- \(x_{ij}\): 이진 결정 변수 — Fine Bin \(j\)부터 \(i\)까지를 하나의 Coarse Bin으로 합병하면 1
- \(V_{ij}\): 해당 합병 조합의 IV 기여분 (사전 계산된 divergence 행렬)

**제약조건:**

| 제약 | 수학적 의미 | 대응 파라미터 |
|------|-----------|-------------|
| 유일 할당 | 각 Fine Bin은 정확히 하나의 Coarse Bin에 소속 | — |
| 인접성 | 합병은 연속된 Fine Bin끼리만 가능 | — |
| 단조성 | 합병 후 Event Rate가 단조증가/감소 | `monotonic_trend` |
| Bin 수 | 최종 Bin 수의 상한/하한 | `min_n_bins`, `max_n_bins` |
| 최소 샘플 | 각 Coarse Bin의 최소 샘플 비율 | `min_bin_size` |
| p-value | 인접 Coarse Bin 간 통계적 유의성 | `max_pvalue` |

---

## 1.2 Solver 선택

```python
optb = OptimalBinning(
    solver="cp",        # 제약 프로그래밍 (기본값, 권장)
    # solver="mip",     # 혼합정수계획법
)
```

| Solver | 내부 엔진 | 특징 |
|--------|----------|------|
| `"cp"` | Google OR-Tools CP-SAT | 제약 전파(constraint propagation) 기반. Pre-bin 20개 이상일 때 효율적 |
| `"mip"` | Google OR-Tools BOP/CBC | 분기한정법(branch-and-bound). Pre-bin 20개 이하에서 빠름 |

!!! note "실무 권장"
    기본값 `"cp"`를 거의 모든 경우에 사용한다. `"mip"`은 변수 수가 수백 개이고 Pre-bin이 적을 때만 고려.
