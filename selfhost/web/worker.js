// worker.js — runs one scc-compiled program for scc.js run() (see exec.js).
import { execute } from './exec.js';

const isNode = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
if (isNode) {
  const { parentPort } = await import('node:worker_threads');
  parentPort.on('message', (msg) => execute(msg, (m) => parentPort.postMessage(m)));
} else {
  self.onmessage = (e) => execute(e.data, (m) => self.postMessage(m));
}
