# Register-Tiled SGEMM — Parameter Walkthrough

This kernel computes `C = A · B` where `A` is `M × K`, `B` is `K × N`, `C` is `M × N`.

The performance comes from a **two-level tiling hierarchy**: each block computes a tile of `C`, and within that tile each thread computes a smaller sub-tile.

---

## The five template parameters

```cpp
template <int BM, int BN, int BK, int TM, int TN>
```

With our values: `BM=128, BN=128, BK=8, TM=8, TN=8`.

| Param | Value | Meaning |
|------|------|---------|
| `BM` | 128 | Block-tile rows of C — one block computes 128 rows |
| `BN` | 128 | Block-tile cols of C — one block computes 128 cols |
| `BK` | 8   | K-dimension chunk loaded per iteration |
| `TM` | 8   | Thread-tile rows — one thread computes 8 rows of C |
| `TN` | 8   | Thread-tile cols — one thread computes 8 cols of C |

---

## The two-level hierarchy

### Level 1: Block tile

The full output `C` is `M × N = 2048 × 2048`. We split it into `BM × BN = 128 × 128` tiles. Each CUDA block is responsible for **one tile**.

```
       N = 2048
   ┌────┬────┬────┬───┐
   │ B  │ B  │ B  │ … │
M  ├────┼────┼────┼───┤
=  │ B  │ B  │ B  │ … │  Each B = one block tile = 128 x 128
2048├────┼────┼────┼───┤
   │ …  │ …  │ …  │ … │
   └────┴────┴────┴───┘

Number of blocks = (2048/128) × (2048/128) = 16 × 16 = 256 blocks
```

### Level 2: Thread tile

Inside each `128 × 128` block tile, we further split into `TM × TN = 8 × 8` thread tiles. Each thread computes one of these.

```
        BN = 128
     ┌──┬──┬──┬─────┐
     │t │t │t │ ... │
BM   ├──┼──┼──┼─────┤
=128 │t │t │t │ ... │   Each t = one thread tile = 8 x 8
     ├──┼──┼──┼─────┤
     │..│..│..│ ... │
     └──┴──┴──┴─────┘

Threads per block = (128/8) × (128/8) = 16 × 16 = 256 threads
```

So **one thread owns 64 elements of C** (`TM × TN = 8 × 8`), held in a register array `c_reg[8][8]`.

---

## Why we sweep K in chunks of `BK`

To compute one C tile, a block needs **all** of:
- A's `BM` rows × full `K` columns
- B's full `K` rows × `BN` columns

That's `128 × 2048 + 2048 × 128 = 524288` floats = 2 MB per block. Way too big for shared memory (~100 KB per SM).

**Solution: stream along K in chunks of `BK` = 8.**

Each iteration loads only:
- `BM × BK = 128 × 8` slice of A → 1024 floats
- `BK × BN = 8 × 128` slice of B → 1024 floats

Total smem per block: `(1024 + 1024) × 4 bytes = 8 KB`. Fits comfortably.

```
   K-step 0:           K-step 1:           K-step 2:    ...
   ┌──┐                  ┌──┐                ┌──┐
   │A0│ × ────┐          │A1│ × ────┐        │A2│ × ────┐
   └──┘       │          └──┘       │        └──┘       │
              ▼                     ▼                   ▼
            ┌───┐                 ┌───┐               ┌───┐
            │B0 │                 │B1 │               │B2 │
            └───┘                 └───┘               └───┘
              │                     │                   │
              └─── add to c_reg ────┴─── add to c_reg ──┴─── ...

Number of K-steps = K / BK = 2048 / 8 = 256
```

After 256 K-steps, `c_reg` holds the final result for that thread's 8×8 tile.

---

## Cooperative loading — what `aLoadsPerThread` means

Per K-step, the block must bring 1024 floats of A into shared memory `s_A`. There are 256 threads. **Each thread handles a fair share**:

```
aLoadsPerThread = (BM × BK) / threadsPerBlock = 1024 / 256 = 4
```

So each thread fetches 4 elements per K-step. Same for B (`bLoadsPerThread = 4`).

### The loading loop

```cpp
for (int i = 0; i < aLoadsPerThread; ++i) {  // 4 iterations
    int idx = i * threadsPerBlock + tid;      // global linear index
    int r   = idx / BK;                       // row inside s_A
    int c   = idx % BK;                       // col inside s_A
    int gr  = blockRow + r;                   // row in global A
    int gc  = t * BK + c;                     // col in global A
    s_A[r][c] = A[gr * K + gc];
}
```

The trick: each iteration, all 256 threads collectively load 256 elements (one per thread), so 4 iterations cover all 1024 elements of `s_A` exactly once.

### Walkthrough for `tid = 5`

| iter `i` | `idx = i*256 + 5` | `r = idx/8` | `c = idx%8` | writes `s_A[r][c]` |
|---|---|---|---|---|
| 0 | 5   | 0  | 5 | `s_A[0][5]`  |
| 1 | 261 | 32 | 5 | `s_A[32][5]` |
| 2 | 517 | 64 | 5 | `s_A[64][5]` |
| 3 | 773 | 96 | 5 | `s_A[96][5]` |

Thread 5 loads 4 elements, evenly spaced 32 rows apart in `s_A`. Across all 256 threads × 4 iterations = 1024 unique `(r, c)` pairs covering the entire tile.

### Why the divisibility matters

`(BM × BK) / threadsPerBlock` **must be an integer** — otherwise some elements get loaded twice or skipped. With our shape, `1024 / 256 = 4` ✓.

If you changed parameters, e.g. `BM=130`, you'd get `130 × 8 / 256 = 4.0625` — non-integer, kernel breaks.

---

## The compute phase (after the load)

Once `s_A` and `s_B` are populated, every thread runs:

```cpp
for (int k = 0; k < BK; ++k) {           // 8 iterations
    for (int i = 0; i < TM; ++i)         // pull a column of s_A into registers
        a_reg[i] = s_A[threadRow + i][k];
    for (int j = 0; j < TN; ++j)         // pull a row of s_B into registers
        b_reg[j] = s_B[k][threadCol + j];
    for (int i = 0; i < TM; ++i)         // outer product: 64 FMAs
        for (int j = 0; j < TN; ++j)
            c_reg[i][j] += a_reg[i] * b_reg[j];
}
```

Per K-step, each thread does:
- 8 reads from `s_A`
- 8 reads from `s_B`
- **64 FMAs**

That's an arithmetic intensity of **4 FMAs per smem read** — high enough that the kernel is no longer memory-bound, which is why it hits ~12 TFLOP/s vs. ~2.5 TFLOP/s for the naive and basic-tiled versions.

---

## Resource budget summary

| Resource | Per block | Notes |
|---|---|---|
| Threads      | 256                           | 16 × 16 layout |
| Shared mem   | 8 KB                          | `s_A` + `s_B` |
| Registers/thread | ~80 floats                | `c_reg[8][8]` + `a_reg[8]` + `b_reg[8]` |
| Outputs/thread | 64                          | `TM × TN` |
| FMAs/thread/K-step | 64                      | Outer product |
| smem reads/thread/K-step | 16                | 8 from A + 8 from B |
| Arithmetic intensity | 4 FMAs/smem-read      | High → compute-bound |

---

## How to think about tuning

The tradeoffs:

- **Bigger `TM`/`TN`** → more reuse per smem read, more FMAs in flight. But also more registers per thread → fewer threads can fit on an SM (lower occupancy) → eventually spills to local memory and tanks performance. Sweet spot is usually 4×4 to 8×8.
- **Bigger `BM`/`BN`** → more work per block, more reuse of each shared tile. But fewer blocks total → may not fill the GPU on small problems.
- **Bigger `BK`** → fewer K-step iterations, less loop overhead. But more smem per block → fewer concurrent blocks per SM. `BK=8` to `BK=16` is typical.

Real cuBLAS tunes these per-shape, per-architecture. For this kernel, `(128, 128, 8, 8, 8)` is a known-good shape that works well on most NVIDIA GPUs from Pascal onwards.
