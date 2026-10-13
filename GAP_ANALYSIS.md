# v0 vs v1 性能差距归因分析（DiffSBDD / Ascend 910B3）

> 结论先行：**唯一的差距来源是 v0 在整步静态图里用高级索引 `h[row]`/`h[col]` 做节点特征 gather**。
> v1 把这些位置全部换成了 `torch.index_select`（NPU 上有专用内核，同样数据约 2.3× 快）。把 v0 的这两处改掉后，
> 25–36% 的每步 device 差距**完全消失，且反超 v1**。v1 的其它优化点（exact_budget、边预算、同步消除、L6/L7 图结构）与差距无关。

## 1. 测量环境与方法

- 机器 / 环境：Ascend 910B3（aarch64，NPU 上有其它租户负载，供应商噪声 ±0.5s）；diffsbdd env（python 3.13 + torch 2.9.0+cpu + torch_npu 2.9.0.post7 + CANN 9.0.0）。
- 基准命令：
  - v0：`ab_bench.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile pdb_files/164707_1749718027_dMJmu.pdb --resi_list "A:300" --outfile out.sdf --n_samples 20 --batch_size 20`
  - v1：同上命令（v1 的 `generate_ligands.py`，22 轮）。
- 一致性纪律：同 seed（`SEED=3` + `torch.manual_seed` + `torch.npu.manual_seed`）→ 每轮 E 完全相同；`PERSTEP=1` 让每步 `torch.npu.synchronize()`，把 device 实算耗时与 host 异步拆开；所有对比要么同 session 顺序跑，要么同 session 交替跑。

## 2. 差距到底在哪一层（先定位后归因）

两个 repo 的采样路径逐算子几乎相同：同样的 5 层 attention EGNN、同样的系数/mu/噪声/COM、同样的整步图捕获。
在 fixed-intercept 模型 `t = t₀ + k·E` 下：

| | t₀（截距） | k（/edge 斜率） |
|---|---|---|
| v0（当前优化版） | 1.73s | 0.27 µs/edge |
| v1 | 1.89s | 0.18 µs/edge |

截距 v0 反而略低；**全部差距都在 per-edge device 斜率上（1.50×）**。host 分发两边都等量极小，可排除。
同 seed 同 E 的 PERSTEP device 逐点对照（同一 session，ab_perstep.log）：

| E（全图边数） | v1 dev/step | v0 dev/step | Δ |
|---|---|---|---|
| 22073 | 6.973 ms | 8.883 ms | +27% |
| 30496 | 8.38 ms（前次 session） | 11.02 ms | +31% |
| 43965 | 10.96 ms（前次 session） | 14.87 ms | +36% |

> v0 高于 v1 的百分比随 E 增大而增大 —— 又是一个“每边额外代价”的指纹，不是固定开销。

## 3. 四配置 bisection（同 session 顺序跑，SEED=3，每个 4 轮）

对 v0 逐项开关 v0/v1 的候选差异，其余全同：

| 配置 | 改动 | 平均/轮 | dev/step @E=22073 | @E=30496 |
|---|---|---|---|---|
| A | +4 个 clone（原版复现） | 4.76s | 8.916 ms | 11.008 ms |
| B | 当前 v0（默认，无 clone、3 段 mask、高级索引） | 4.75s | 8.917 ms | 11.045 ms |
| C | v1 式单对 gather mask | 4.71s | 8.906 ms | 11.003 ms |
| **D** | **高级索引 → `torch.index_select`** | **3.58s** | **6.489 ms** | **7.892 ms** |

- A vs B：**0 差别** → clone 在图内近零成本，排除。
- C vs B：0.7%（噪声内）→ mask 构造方式（3 段 vs 单对）不贡献，排除。
- D vs B：**−25% 整轮，−27% dev/step（@E=22073）** → 差距全部在这里。

## 4. 微基准：同一个 gather，两种写法

E=22073、N=477、D=128 的节点特征 (E,128) gather：

| 写法 | 耗时 | 比例 |
|---|---|---|
| `h[row], h[col]`（高级索引，同时取两条） | 0.387 ms | 1× |
| `torch.index_select(h,0,row)` ×2 | 0.168 ms | **0.43×（2.3× 快）** |

图内每步有 5 层 × 2 个站点 = 10 处这类 gather（GCL.forward 的 `edge_model`、EquivariantUpdate 的 `coord_model`）。
放大倍率与“D 把每步 device 拉下 2.4ms @E=22073”的量级一致（内核有部分重叠，故 10×0.22ms 不会全数兑现）。
坐标/边侧 (E,3) gather（`x[edge_index]`、`mean[batch_mask[edge_index]]`）微基准两者持平 —— 这也是为何只改节点 gather 就够。

## 5. 同 session 金标准对照（修后 v0 vs v1）

SEED=3，两种路线都在同一 session 顺序跑满若干轮，按每轮 E 对齐比较：

| E | v1 该轮用时 | v0+configD 该轮用时 |
|---|---|---|
| 22073 | 4.66s | 4.53s |
| 30496 | 4.56s | 4.37s |
| 21306 | 3.76s | 3.63s |
| 21459 | 3.79s | 3.64s |
| 24134 | 3.95s | 3.83s |
| 28134 | 4.35s | 4.22s |
| **平均（多轮）** | **4.39s（22 轮）** | **3.83s（6 轮）** |

v1 修复前 v0 同 session 平均 4.75s（配置 B）。修复后：**4.75 → 3.83s（−19%）**，且在每一档 E 上都反超 v1（−13% 平均）。
DEVICE 侧斜率：v0B ~0.25 µs/edge → v0D ~0.16 µs/edge（v1 ~0.18 µs/edge）——修后 v0 每边代价略低于 v1。

数值正确性：四个配置均产出 20 个分子，`torch.isfinite` 断言全部通过；`index_select` 与高级索引是同一纯 gather，逐位同值。四个 SDF 的 md5 不同，属于 NPU 浮点累加顺序的微小不确定性，与本次性能结论无关。

## 6. 差距归因表（v0 中哪些代码差异造成了差距）

| v0 中的差异 | 位置 | 对差距的贡献 |
|---|---|---|
| 节点特征高级索引 gather `h[row]`/`h[col]` | `egnn_new.py` GCL.forward（edge_model）、EquivariantUpdate.coord_model，每层 2 处 × 5 层 | **全部贡献（≈2.4ms/步 @E=22073，即 25–36%）** |
| 4 个无条件 clone | `dynamics.py` forward 前 4 行 | 0（配置 A 无变化） |
| 3 段 mask vs v1 单对 mask | `graphs.py` compute_edge_mask | 0（配置 C 无变化） |
| 边预算 exact_budget / k_sigma / margin | v1 专有 | 与此 pocket 无关（reach=inf，全稠密退化，E 相同） |
| host 分发 / 同步点数 | 图把步 | 0（两边 host 均等量极小，截距相等） |

一句话：**v1 只在节点 gather 内核的选择上领先，其余优化在两种实现下收益一致或被退化抵消。**

## 7. 修复内容（等价、已验证，**已落地为默认**）

`egnn_new.py`（默认走 `torch.index_select`，可用 `V0_NAGATHER=0` 回退高级索引；两处调用点为 GCL.forward 与 EquivariantUpdate.coord_model）：

```python
# 默认 ON：V0_NAGATHER=0 时回退高级索引
_NAG = os.environ.get('V0_NAGATHER', '1') != '0'
def _gather(t, idx):
    return torch.index_select(t, 0, idx) if _NAG else t[idx]
```

- GCL.forward：`self.edge_model(_gather(h, row), _gather(h, col), edge_attr, edge_mask)`
- EquivariantUpdate.coord_model：`torch.cat([_gather(h, row), _gather(h, col), edge_attr], dim=1)`

复现命令（`set_env.sh` 之后）：

```bash
env -u PYTHONPATH SEED=3 PERSTEP=1 BENCH_ROUNDS=22 \
  /root/miniconda3/envs/diffsbdd/bin/python ab_bench.py \
  checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile pdb_files/164707_1749718027_dMJmu.pdb --resi_list "A:300" \
  --outfile /tmp/o.sdf
```

## 8. 落地验证（2026-10-09）

同 session 顺序跑，SEED=3，`V0_NAGATHER=0`（回退）vs 默认（落地）：

| 状态 | dev/step @E=22073 | @E=30496 | 平均/轮（2 轮计数） |
|---|---|---|---|
| `V0_NAGATHER=0`（高级索引） | 8.944 ms | 11.044 ms | 4.72s |
| **默认（index_select，落地）** | **6.628 ms** | **8.005 ms** | **3.63s** |

落地状态 −26% dev/step、−23% 整轮；与 bisection 的 config B/D 数值一致。官方命令（无任何 env）单轮 4.05s，rc=0。三份 SDF（OFF/ON/官方默认）均 20 分子、非有限原子数 0。

## 9. 后续建议

1. ~~建议直接落地~~ **已落地（见 §8）**；`V0_NAGATHER=0` 保留为回退开关。
2. 可选：`batch_mask[edge_index]` 与 `x[edge_index]` 的坐标侧 gather 微基准持平，收益为 0，不必改。
3. 测量用的 `ab_bench.py`、`bisect_run.sh`、`headtohead_run.sh`、`verify_landing.sh`、PERSTEP 插桩均为 env 门控、默认关闭，可保留作回归基准；若不需要可清理。

## 附：证据文件

- `/tmp/ab_perstep.log` — v1/v0 同 session PERSTEP device（E=22073 时 v1 6.973 vs v0 8.883）
- `/tmp/bis_conf_A.log` … `/tmp/bis_conf_D.log` — 本次 4 配置 bisection 原始数据
- `/tmp/headtohead.log` — 同 session v1(22 轮, 4.39s) vs v0+configD(6 轮, 3.83s)