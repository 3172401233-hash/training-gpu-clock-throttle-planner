/**
 * GPU 降频处理规划器（纯函数，不访问真实 GPU / nvidia-smi）。
 *
 * 输入：GPU 时钟/温度/功耗、任务阶段、checkpoint、备用节点、通信拓扑。
 * 输出：受影响任务、限频/迁移/暂停顺序、迁移批次、预计吞吐、不可迁移原因。
 */

export const DEFAULT_OPTIONS = {
  clockThrottleRatio: 0.85, // 时钟低于基准的该比例视为降频
  mildThrottleRatio: 0.7, // 高于该比例视为轻度降频，可就地限频
  capClockRatio: 0.75, // 限频动作将时钟限制到基准的该比例
  recoveryWarmupSteps: 20, // 迁移后性能恢复的预热 step 数
  recoveryThroughputFactor: 0.95, // 预热期内吞吐折扣
  migrationBatchSize: 2, // 每批迁移的任务数，避免网络/存储风暴
  pausePhases: ['checkpointing'], // 这些阶段禁止迁移，只能等待或暂停
};

const PHASE_MIGRATABLE = new Set(['forward', 'backward', 'optimizer', 'idle', 'evaluation']);

/** 检测单张 GPU 的降频状态。 */
export function detectGpuThrottle(gpu, options = DEFAULT_OPTIONS) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const ratio = gpu.clockMHz / gpu.baseClockMHz;
  const reasons = [];
  if (ratio < opts.clockThrottleRatio) reasons.push('clock');
  if (gpu.tempC >= gpu.tempLimitC) reasons.push('thermal');
  if (gpu.powerW >= gpu.powerLimitW) reasons.push('power');
  return {
    id: gpu.id,
    node: gpu.node,
    clockRatio: round4(ratio),
    throttled: reasons.length > 0,
    mild: reasons.length > 0 && ratio >= opts.mildThrottleRatio,
    reasons,
  };
}

/** 汇总节点级事件：单卡热点 vs 整节点降频。 */
export function detectNodeEvents(gpus, options = DEFAULT_OPTIONS) {
  const statuses = gpus.map((gpu) => detectGpuThrottle(gpu, options));
  const byNode = new Map();
  for (const status of statuses) {
    if (!byNode.has(status.node)) byNode.set(status.node, []);
    byNode.get(status.node).push(status);
  }
  const events = [];
  for (const [node, list] of byNode) {
    const throttled = list.filter((s) => s.throttled);
    if (throttled.length === 0) continue;
    const allMild = throttled.every((s) => s.mild);
    events.push({
      node,
      type: throttled.length >= 2 ? 'node-wide' : 'single-gpu',
      severity: allMild ? 'mild' : 'severe',
      gpus: throttled,
    });
  }
  return { gpuStatuses: statuses, nodeEvents: events };
}

/** 校验 checkpoint 完整性。 */
export function checkCheckpointIntegrity(checkpoint) {
  if (!checkpoint) return { ok: false, reason: 'checkpoint 缺失' };
  if (checkpoint.complete !== true) return { ok: false, reason: 'checkpoint 不完整' };
  if (!Number.isFinite(checkpoint.step) || checkpoint.step < 0) {
    return { ok: false, reason: 'checkpoint step 非法' };
  }
  if (checkpoint.shardsExpected != null) {
    const present = checkpoint.shardsPresent ?? 0;
    if (present < checkpoint.shardsExpected) {
      return { ok: false, reason: `checkpoint 分片缺失 ${present}/${checkpoint.shardsExpected}` };
    }
  }
  return { ok: true, reason: null };
}

/**
 * 估算集合通信下的任务吞吐（相对健康基线 1.0）。
 * 集合通信全局 step 由最慢 rank 决定：step 时间 = max(1/ratio_i)。
 */
export function estimateThroughput(rankRatios) {
  if (rankRatios.length === 0) return 1;
  const slowest = Math.min(...rankRatios);
  return round4(Math.max(slowest, 0));
}

/** 通信拓扑感知的迁移目标选择：尽量把任务 ranks 集中到同一备用节点。 */
function allocateSpares(task, spareNodes) {
  const needed = task.ranks.length;
  const sorted = [...spareNodes].sort((a, b) => b.freeGpus - a.freeGpus);
  const allocation = [];
  let remaining = needed;
  for (const spare of sorted) {
    if (remaining === 0) break;
    if (spare.freeGpus === 0) continue;
    const take = Math.min(spare.freeGpus, remaining);
    allocation.push({ node: spare.node, gpus: take });
    remaining -= take;
  }
  return remaining === 0 ? allocation : null;
}

function decideTask(task, eventByNode, gpuStatusById, spareNodes, options) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const rankRatios = task.ranks.map((rank) => gpuStatusById.get(rank.gpu)?.clockRatio ?? 1);
  const currentThroughput = estimateThroughput(rankRatios);
  const impactedNodes = [...new Set(
    task.ranks
      .filter((rank) => gpuStatusById.get(rank.gpu)?.throttled)
      .map((rank) => gpuStatusById.get(rank.gpu).node),
  )];
  const events = impactedNodes.map((node) => eventByNode.get(node)).filter(Boolean);
  const severe = events.some((event) => event.severity === 'severe' || event.type === 'node-wide');

  const base = {
    taskId: task.id,
    phase: task.phase,
    impactedNodes,
    currentThroughput,
    action: null,
    migration: null,
    nonMigratableReason: null,
    recovery: null,
  };

  // 阶段约束：checkpointing 等阶段不可迁移
  if (!PHASE_MIGRATABLE.has(task.phase) || opts.pausePhases.includes(task.phase)) {
    return {
      ...base,
      action: 'pause',
      nonMigratableReason: `任务处于 ${task.phase} 阶段，禁止迁移`,
      throughputAfter: 0,
    };
  }

  // 轻度单卡降频：就地限频，保住节点健康，避免迁移开销
  if (!severe) {
    return {
      ...base,
      action: 'cap-clock',
      capClockRatio: opts.capClockRatio,
      throughputAfter: round4(Math.min(currentThroughput / Math.min(...rankRatios), 1) * opts.capClockRatio),
    };
  }

  // 严重/整节点事件：优先迁移，需要完整 checkpoint 与足够备用节点
  const integrity = checkCheckpointIntegrity(task.checkpoint);
  if (!integrity.ok) {
    return {
      ...base,
      action: 'pause',
      nonMigratableReason: `${integrity.reason}，迁移将丢失激活，暂停等待 checkpoint`,
      throughputAfter: 0,
    };
  }
  const allocation = allocateSpares(task, spareNodes);
  if (!allocation) {
    return {
      ...base,
      action: 'pause',
      nonMigratableReason: '备用节点不足',
      throughputAfter: 0,
    };
  }
  return {
    ...base,
    action: 'migrate',
    migration: {
      fromNodes: impactedNodes,
      toNodes: allocation,
      checkpointStep: task.checkpoint.step,
    },
    recovery: {
      warmupSteps: opts.recoveryWarmupSteps,
      throughputDuringWarmup: opts.recoveryThroughputFactor,
      throughputAfterRecovery: 1,
    },
    throughputAfter: opts.recoveryThroughputFactor,
  };
}

/**
 * 主入口：计算限频、迁移、暂停的全局顺序。
 * 顺序原则：先就地限频（零开销止损）→ 分批迁移 → 最后暂停。
 */
export function planThrottleResponse(input, options = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const { gpuStatuses, nodeEvents } = detectNodeEvents(input.gpus, opts);
  const gpuStatusById = new Map(gpuStatuses.map((s) => [s.id, s]));
  const eventByNode = new Map(nodeEvents.map((e) => [e.node, e]));
  const spareNodes = (input.spareNodes ?? []).map((s) => ({ ...s }));

  const affectedTasks = [];
  for (const task of input.tasks ?? []) {
    const touched = task.ranks.some((rank) => gpuStatusById.get(rank.gpu)?.throttled);
    if (!touched) continue;
    const decision = decideTask(task, eventByNode, gpuStatusById, spareNodes, opts);
    // 迁移成功后扣减备用节点容量，保证批次内不超卖
    if (decision.action === 'migrate') {
      for (const target of decision.migration.toNodes) {
        const spare = spareNodes.find((s) => s.node === target.node);
        spare.freeGpus -= target.gpus;
      }
    }
    affectedTasks.push(decision);
  }

  const migrations = affectedTasks.filter((t) => t.action === 'migrate');
  const migrationBatches = [];
  for (let i = 0; i < migrations.length; i += opts.migrationBatchSize) {
    migrationBatches.push({
      batch: migrationBatches.length + 1,
      tasks: migrations.slice(i, i + opts.migrationBatchSize).map((t) => ({
        taskId: t.taskId,
        ...t.migration,
      })),
    });
  }

  const actionOrder = [
    ...affectedTasks.filter((t) => t.action === 'cap-clock').map((t) => ({ order: 'cap-clock', taskId: t.taskId })),
    ...migrationBatches.flatMap((batch) =>
      batch.tasks.map((t) => ({ order: `migrate-batch-${batch.batch}`, taskId: t.taskId })),
    ),
    ...affectedTasks.filter((t) => t.action === 'pause').map((t) => ({ order: 'pause', taskId: t.taskId })),
  ];

  const nonMigratable = affectedTasks
    .filter((t) => t.nonMigratableReason)
    .map((t) => ({ taskId: t.taskId, reason: t.nonMigratableReason }));

  const healthyTasks = (input.tasks ?? []).filter(
    (task) => !affectedTasks.some((t) => t.taskId === task.id),
  ).length;
  const estimatedThroughput = round4(
    healthyTasks + affectedTasks.reduce((sum, t) => sum + (t.throughputAfter ?? 0), 0),
  );

  return {
    nodeEvents,
    affectedTasks,
    actionOrder,
    migrationBatches,
    estimatedThroughput,
    nonMigratable,
  };
}

function round4(value) {
  return Math.round(value * 10000) / 10000;
}
