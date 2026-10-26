// exec.js — runs one scc-compiled program: execute({ wasm, stdin, args,
// maxMemoryMb }, post) posts {type:'stdout'|'stderr', data} as output arrives
// and finally {type:'exit', code}, and resolves to the exit code. Used by
// worker.js, and by scc.js directly where no Worker can be started.
class Exit extends Error { constructor(code) { super('exit'); this.code = code; } }

// sets the maximum of the module's memory (section 5) to 'pages'
function limitMemory(bytes, pages) {
  const b = bytes;
  const out = [];
  let i = 8;
  const leb = () => { let r = 0, s = 0, x; do { x = b[i++]; r |= (x & 127) << s; s += 7; } while (x & 128); return r >>> 0; };
  const enc = (v) => { const a = []; do { let x = v & 127; v >>>= 7; if (v) x |= 128; a.push(x); } while (v); return a; };
  out.push(...b.subarray(0, 8));
  while (i < b.length) {
    const id = b[i++];
    const start = i;
    const size = leb();
    const body = i;
    if (id === 5) {
      let j = i;
      const n = (() => { let r = 0, s = 0, x; do { x = b[j++]; r |= (x & 127) << s; s += 7; } while (x & 128); return r; })();
      if (n === 1) {
        const flags = b[j++];
        i = j;
        const min = leb();
        if (flags & 1) leb();
        const sec = [1, 1, ...enc(min), ...enc(Math.max(pages, min))];
        out.push(5, ...enc(sec.length), ...sec);
        i = body + size;
        continue;
      }
    }
    out.push(id, ...b.subarray(start, body + size));
    i = body + size;
  }
  return new Uint8Array(out);
}

export async function execute(msg, post) {
  const enc = new TextEncoder();
  const dec1 = new TextDecoder();
  const dec2 = new TextDecoder();
  const input = enc.encode(msg.stdin || '');
  let inPos = 0;
  let memory = null;
  let code = 0;
  const imports = {
    env: {
      __host_write(fd, ptr, len) {
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        if (fd === 2) post({ type: 'stderr', data: dec2.decode(bytes, { stream: true }) });
        else post({ type: 'stdout', data: dec1.decode(bytes, { stream: true }) });
        return len;
      },
      __host_read(fd, ptr, len) {
        const n = Math.min(len, input.length - inPos);
        new Uint8Array(memory.buffer, ptr, n).set(input.subarray(inPos, inPos + n));
        inPos += n;
        return n;
      },
      __host_exit(c) { throw new Exit(c); },
    },
  };
  try {
    const pages = Math.max(1, Math.floor((msg.maxMemoryMb || 256) * 16));
    const { instance } = await WebAssembly.instantiate(limitMemory(msg.wasm, pages), imports);
    memory = instance.exports.memory;
    const main = instance.exports.main;
    if (main.length >= 2) {
      const base = instance.exports.__rt_argv();
      const view = new DataView(memory.buffer);
      let at = base + 4 * (msg.args.length + 1);
      msg.args.forEach((a, k) => {
        const s = enc.encode(a + '\0');
        new Uint8Array(memory.buffer, at, s.length).set(s);
        view.setInt32(base + 4 * k, at, true);
        at += s.length;
      });
      view.setInt32(base + 4 * msg.args.length, 0, true);
      code = main(msg.args.length, base);
    } else {
      code = main();
    }
  } catch (e) {
    if (e instanceof Exit) code = e.code;
    else if (e instanceof WebAssembly.RuntimeError) {
      const m = String(e.message);
      code = /divide by zero|integer overflow/.test(m) ? 136 : /call stack|stack overflow/i.test(m) ? 139 : 134;
      post({ type: 'stderr', data: `\n[trap: ${m}]\n` });
    } else if (e instanceof RangeError && /memory/i.test(String(e.message))) {
      code = 137;
      post({ type: 'stderr', data: '\n[memory limit exceeded]\n' });
    } else {
      code = 1;
      post({ type: 'stderr', data: `\n[${e && e.message || e}]\n` });
    }
  }
  post({ type: 'exit', code: code & 255 });
  return code & 255;
}
