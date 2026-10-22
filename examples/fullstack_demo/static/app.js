// fullstack_demo — loads counter_client.wasm through std's DOM glue
// (safec_dom.js is a copy of <SafeC>/std/wasm/safec_dom.js) and hydrates
// the server-rendered /counter page.
import { loadSafeC } from "/static/safec_dom.js";

const app = await loadSafeC(fetch("/static/counter_client.wasm"), { prefix: "[counter_client] " });
app.exports.hydrate_init();
