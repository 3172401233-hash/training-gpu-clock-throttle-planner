import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  planThrottleResponse,
  detectGpuThrottle,
  checkCheckpointIntegrity,
  estimateThroughput,
} from '../src/planner.js';

const healthyGpu = (id, node) => ({
  id,
  node,
  clockMHz: 1980,
  baseClockMHz: 1980,
  tempC: 60,
  tempLimitC: 87,
  powerW: 300,
  powerLimitW: 400,
});

test('单卡热点：轻度降频就地限频，不迁移', () => {
  const input = {
    gpus: [
      healthyGpu('g0', 'node-a'),
      { ...healthyGpu('g1', 'node-a'), clockMHz: 1500, tempC: 90 },
    ],
    tasks: [
      {
        id: 't1',
        phase: 'backward',
        ranks: [
          { rank: 0, gpu: 'g0' },
          { rank: 1, gpu: 'g1' },
        ],
        checkpoint: { step: 100, complete: true },
      },
    ],
    spareNodes: [{ node: 'spare-1', freeGpus: 2 }],
    topology: { type: 'ring', order: ['t1:0', 't1:1'] },
  };
  const plan = planThrottleResponse(input);

  assert.equal(plan.nodeEvents.length, 1);
  assert.equal(plan.nodeEvents[0].type, 'single-gpu');
  assert.equal(plan.nodeEvents[0].severity, 'mild');

  const task = plan.affectedTasks.find((t) => t.taskId === 't1');
  assert.equal(task.action, 'cap-clock');
  assert.equal(task.migration, null);
  assert.ok(task.currentThroughput < 1, '集合通信被慢卡拖慢');
  assert.equal(plan.migrationBatches.length, 0);
  assert.deepEqual(plan.actionOrder.map((a) => a.order), ['cap-clock']);
});

test('整节点降频：checkpoint 完整时迁移到备用节点', () => {
  const input = {
    gpus: [
      { ...healthyGpu('g0', 'node-a'), clockMHz: 800, tempC: 96 },
      { ...healthyGpu('g1', 'node-a'), clockMHz: 780, tempC: 97 },
      healthyGpu('g2', 'node-b'),
      healthyGpu('g3', 'node-b'),
    ],
    tasks: [
      {
        id: 't1',
        phase: 'backward',
        ranks: [
          { rank: 0, gpu: 'g0' },
          { rank: 1, gpu: 'g1' },
          { rank: 2, gpu: 'g2' },
          { rank: 3, gpu: 'g3' },
        ],
        checkpoint: { step: 1200, complete: true, shardsExpected: 4, shardsPresent: 4 },
      },
    ],
    spareNodes: [{ node: 'spare-1', freeGpus: 4 }],
    topology: { type: 'ring', order: ['t1:0', 't1:1', 't1:2', 't1:3'] },
  };
  const plan = planThrottleResponse(input);

  assert.equal(plan.nodeEvents[0].type, 'node-wide');
  assert.equal(plan.nodeEvents[0].severity, 'severe');

  const task = plan.affectedTasks.find((t) => t.taskId === 't1');
  assert.equal(task.action, 'migrate');
  assert.deepEqual(task.migration.fromNodes, ['node-a']);
  assert.deepEqual(task.migration.toNodes, [{ node: 'spare-1', gpus: 4 }]);
  assert.equal(task.migration.checkpointStep, 1200);

  // 恢复后性能：预热期折扣，之后回到基线
  assert.ok(task.recovery.warmupSteps > 0);
  assert.ok(task.recovery.throughputDuringWarmup < 1);
  assert.equal(task.recovery.throughputAfterRecovery, 1);

  assert.equal(plan.migrationBatches.length, 1);
  assert.equal(plan.migrationBatches[0].tasks[0].taskId, 't1');
  assert.equal(plan.nonMigratable.length, 0);
  assert.ok(plan.estimatedThroughput > task.currentThroughput, '迁移后吞吐回升');
});

test('checkpoint 缺失：不可迁移，暂停并给出原因', () => {
  const input = {
    gpus: [
      { ...healthyGpu('g0', 'node-a'), clockMHz: 700, tempC: 98 },
      { ...healthyGpu('g1', 'node-a'), clockMHz: 700, tempC: 98 },
    ],
    tasks: [
      {
        id: 't-no-ckpt',
        phase: 'forward',
        ranks: [
          { rank: 0, gpu: 'g0' },
          { rank: 1, gpu: 'g1' },
        ],
        checkpoint: null,
      },
      {
        id: 't-bad-shards',
        phase: 'forward',
        ranks: [{ rank: 0, gpu: 'g0' }],
        checkpoint: { step: 50, complete: true, shardsExpected: 4, shardsPresent: 2 },
      },
    ],
    spareNodes: [{ node: 'spare-1', freeGpus: 8 }],
    topology: { type: 'ring', order: [] },
  };
  const plan = planThrottleResponse(input);

  const noCkpt = plan.affectedTasks.find((t) => t.taskId === 't-no-ckpt');
  assert.equal(noCkpt.action, 'pause');
  assert.match(noCkpt.nonMigratableReason, /checkpoint 缺失/);

  const badShards = plan.affectedTasks.find((t) => t.taskId === 't-bad-shards');
  assert.equal(badShards.action, 'pause');
  assert.match(badShards.nonMigratableReason, /分片缺失/);

  assert.equal(plan.nonMigratable.length, 2);
  assert.equal(plan.migrationBatches.length, 0);
  // 暂停排在顺序末尾
  assert.deepEqual(plan.actionOrder.map((a) => a.order), ['pause', 'pause']);
});

test('备用节点不足：不可迁移并暂停', () => {
  const input = {
    gpus: [
      { ...healthyGpu('g0', 'node-a'), clockMHz: 700, tempC: 98 },
      { ...healthyGpu('g1', 'node-a'), clockMHz: 700, tempC: 98 },
    ],
    tasks: [
      {
        id: 't1',
        phase: 'optimizer',
        ranks: [
          { rank: 0, gpu: 'g0' },
          { rank: 1, gpu: 'g1' },
        ],
        checkpoint: { step: 10, complete: true },
      },
    ],
    spareNodes: [],
    topology: { type: 'ring', order: [] },
  };
  const plan = planThrottleResponse(input);
  const task = plan.affectedTasks[0];
  assert.equal(task.action, 'pause');
  assert.equal(task.nonMigratableReason, '备用节点不足');
});

test('checkpointing 阶段禁止迁移', () => {
  const input = {
    gpus: [
      { ...healthyGpu('g0', 'node-a'), clockMHz: 600, tempC: 99 },
      { ...healthyGpu('g1', 'node-a'), clockMHz: 600, tempC: 99 },
    ],
    tasks: [
      {
        id: 't1',
        phase: 'checkpointing',
        ranks: [
          { rank: 0, gpu: 'g0' },
          { rank: 1, gpu: 'g1' },
        ],
        checkpoint: { step: 5, complete: true },
      },
    ],
    spareNodes: [{ node: 'spare-1', freeGpus: 2 }],
    topology: { type: 'ring', order: [] },
  };
  const plan = planThrottleResponse(input);
  assert.equal(plan.affectedTasks[0].action, 'pause');
  assert.match(plan.affectedTasks[0].nonMigratableReason, /checkpointing 阶段/);
});

test('多任务迁移分批，备用节点容量不超卖', () => {
  const makeTask = (id, gpus) => ({
    id,
    phase: 'backward',
    ranks: gpus.map((gpu, rank) => ({ rank, gpu })),
    checkpoint: { step: 1, complete: true },
  });
  const input = {
    gpus: [
      { ...healthyGpu('g0', 'node-a'), clockMHz: 500, tempC: 99 },
      { ...healthyGpu('g1', 'node-a'), clockMHz: 500, tempC: 99 },
      { ...healthyGpu('g2', 'node-b'), clockMHz: 500, tempC: 99 },
      { ...healthyGpu('g3', 'node-b'), clockMHz: 500, tempC: 99 },
    ],
    tasks: [makeTask('t1', ['g0', 'g1']), makeTask('t2', ['g2', 'g3'])],
    spareNodes: [{ node: 'spare-1', freeGpus: 4 }],
    topology: { type: 'ring', order: [] },
  };
  const plan = planThrottleResponse(input, { migrationBatchSize: 1 });
  assert.equal(plan.migrationBatches.length, 2, '两个任务分两批迁移');
  assert.ok(plan.affectedTasks.every((t) => t.action === 'migrate'));
});

test('单元：降频检测、checkpoint 完整性、吞吐估算', () => {
  const status = detectGpuThrottle({
    id: 'g',
    node: 'n',
    clockMHz: 1000,
    baseClockMHz: 2000,
    tempC: 95,
    tempLimitC: 87,
    powerW: 410,
    powerLimitW: 400,
  });
  assert.deepEqual(status.reasons.sort(), ['clock', 'power', 'thermal']);
  assert.equal(status.mild, false);

  assert.equal(checkCheckpointIntegrity(null).ok, false);
  assert.equal(checkCheckpointIntegrity({ step: 1, complete: false }).ok, false);
  assert.equal(checkCheckpointIntegrity({ step: 1, complete: true }).ok, true);

  // 集合通信由最慢 rank 决定
  assert.equal(estimateThroughput([1, 1, 0.5, 1]), 0.5);
  assert.equal(estimateThroughput([1, 1]), 1);
});

test('健康集群：无受影响任务，吞吐为任务数', () => {
  const input = {
    gpus: [healthyGpu('g0', 'node-a'), healthyGpu('g1', 'node-a')],
    tasks: [
      {
        id: 't1',
        phase: 'forward',
        ranks: [{ rank: 0, gpu: 'g0' }],
        checkpoint: { step: 1, complete: true },
      },
    ],
    spareNodes: [],
    topology: { type: 'ring', order: [] },
  };
  const plan = planThrottleResponse(input);
  assert.equal(plan.affectedTasks.length, 0);
  assert.equal(plan.estimatedThroughput, 1);
});
