// safec-doc — loads the wasm client (safeguard/doctool/client, built to
// assets/doc.wasm) through std/wasm's JS glue (assets/safec_dom.js,
// copied from std/wasm/safec_dom.js at build time). The page works
// without it; the client only adds behavior.
import { loadSafeC } from './safec_dom.js';

const imports = {
  // std/wasm has no clipboard binding: the one site-specific import.
  doc_clipboard_write: (ptr, len) => {
    const text = new TextDecoder().decode(new Uint8Array(app.exports.memory.buffer, Number(ptr), Number(len)));
    navigator.clipboard?.writeText(text).catch(() => {});
  },
  // Referenced by std's float formatting, never called by the client.
  snprintf: () => 0,
};

// The glue copies globalThis's members into a plain object and calls them
// as methods of it; browser APIs that need their real receiver are bound.
const globals = {
  fetch: (...a) => fetch(...a),
  requestAnimationFrame: (cb) => requestAnimationFrame(cb),
  cancelAnimationFrame: (id) => cancelAnimationFrame(id),
  setTimeout: (...a) => setTimeout(...a),
  clearTimeout: (id) => clearTimeout(id),
};

let app = null;
try {
  app = await loadSafeC(fetch(new URL('doc.wasm', import.meta.url)), { imports, globals, prefix: '[safec-doc] ' });
  app.exports.doc_init();
} catch (e) {
  console.warn('doc.wasm unavailable; the page works without it.', e);
}
