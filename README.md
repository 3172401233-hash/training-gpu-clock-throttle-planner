# training-gpu-clock-throttle-planner

训练集群 GPU 降频处理规划器：根据 GPU 时钟、温度、功耗、任务阶段、checkpoint、备用节点和通信拓扑，计算**限频、迁移、暂停**的执行顺序，输出受影响任务、迁移批次、预计吞吐和不可迁移原因。

> **说明：本项目是纯计算规划器，不访问真实 GPU。** 不调用 `nvidia-smi`、NVML 或任何硬件接口；所有 GPU 状态（时钟/温度/功耗）以 JSON 输入提供，输出仅为调度建议，不执行真实迁移。

## 环境

- Node.js >= 24.16.0（无第三方依赖）

## 使用

```bash
node src/cli.js examples/cluster.json
```

输入 JSON 字段：

| 字段 | 说明 |
| --- | --- |
| `gpus[]` | `id`、`node`、`clockMHz`、`baseClockMHz`、`tempC`、`tempLimitC`、`powerW`、`powerLimitW` |
| `tasks[]` | `id`、`phase`（forward/backward/optimizer/checkpointing 等）、`ranks[]`（rank → gpu 映射）、`checkpoint`（`step`、`complete`、可选 `shardsExpected`/`shardsPresent`） |
| `spareNodes[]` | `node`、`freeGpus` |
| `topology` | 通信拓扑（如 `ring` 及 rank 顺序），用于感知集合通信中最慢 rank 拖长全局 step |

输出 JSON 字段：

| 字段 | 说明 |
| --- | --- |
| `nodeEvents` | 降频事件分类：`single-gpu`（单卡热点）/ `node-wide`（整节点降频），含严重度 |
| `affectedTasks` | 每个受影响任务的动作（`cap-clock` / `migrate` / `pause`）、当前吞吐、迁移目标、恢复后性能 |
| `actionOrder` | 全局执行顺序：先就地限频止损 → 分批迁移 → 最后暂停 |
| `migrationBatches` | 迁移批次（默认每批 2 个任务，避免网络/存储风暴；备用节点容量不超卖） |
| `estimatedThroughput` | 集群预计吞吐（相对健康基线，集合通信 step 由最慢 rank 决定） |
| `nonMigratable` | 不可迁移任务及原因（checkpoint 缺失/不完整、备用节点不足、阶段禁止迁移） |

## 决策逻辑

1. **单卡轻度降频**（时钟 ≥ 基准 70%）：就地限频到基准 75%，保住节点健康，避免迁移开销。
2. **整节点或严重降频**：checkpoint 完整且备用节点充足时迁移；迁移后给出恢复预热期（默认 20 step，预热吞吐 95%）与恢复后性能。
3. **不可迁移则暂停**：checkpoint 缺失/分片不全（迁移会丢失激活）、备用节点不足、或任务处于 `checkpointing` 等禁止迁移阶段。

## 测试

```bash
node --test
```

覆盖：单卡热点、整节点降频、checkpoint 缺失/分片不全、备用节点不足、checkpointing 阶段禁迁、多任务分批迁移、吞吐估算与健康集群基线。
