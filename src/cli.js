#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { planThrottleResponse } from './planner.js';

const usage = '用法: node src/cli.js <input.json>\n输入为集群状态 JSON（GPU 时钟/温度/功耗、任务、checkpoint、备用节点、拓扑）。';

const file = process.argv[2];
if (!file) {
  console.error(usage);
  process.exit(1);
}

try {
  const input = JSON.parse(await readFile(file, 'utf8'));
  const plan = planThrottleResponse(input);
  console.log(JSON.stringify(plan, null, 2));
} catch (error) {
  console.error(`规划失败: ${error.message}`);
  process.exit(1);
}
