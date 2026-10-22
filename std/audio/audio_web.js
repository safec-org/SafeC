// SafeC Standard Library — std/audio's WebAssembly host glue: Web Audio and
// Web MIDI for a wasm32 module built with std/audio (audio_webaudio_wasm.sc,
// midi_webmidi_wasm.sc, audio_sys.sc's clock).
//
// A wasm32-unknown-unknown module has no audio or MIDI access of its own;
// the externs those files declare become imports from the "env" module,
// and this file supplies them. Use it as a classic <script>, an ES module
// side import, or require() it under Node; it defines globalThis.SafeCAudio:
//
//   const env = SafeCAudio.imports({ /* the module's other imports */ });
//   const { instance } = await WebAssembly.instantiate(bytes, { env });
//   SafeCAudio.attach(instance);           // before calling into the module
//   await SafeCAudio.enableMIDI({ sysex: false });   // optional: Web MIDI ports
//   await SafeCAudio.enableInput();        // optional: microphone as an input device
//   button.onclick = () => { instance.exports.start_audio(); SafeCAudio.resume(); };
//
// Streams run on an AudioWorklet when the browser has one — the worklet
// plays from a small queue of periods the main thread renders through the
// module on request (its MessagePort) — and fall back to a
// ScriptProcessorNode (onaudioprocess calls the module directly). Set
// SafeCAudio.options.worklet = false to force the fallback. Browsers start
// audio only after a user gesture: open/start streams from an event
// handler, or call SafeCAudio.resume() from one.
//
// SafeCAudio.options.AudioContext and .navigator can be replaced (tests
// run this file under Node with stand-ins for both).
(function (root) {
    "use strict";

    const options = {
        worklet: true,
        AudioContext: root.AudioContext || root.webkitAudioContext || null,
        navigator: root.navigator || null,
        now: () => (root.performance ? root.performance.now() : Date.now()),
    };

    let instance = null;
    const streams = [];          // handle -> stream state
    const midi = { access: null, inputs: [], outputs: [], open: [] };
    let inputStream = null;      // MediaStream from enableInput()
    let inputChannels = 0;

    const exp = () => {
        if (!instance) throw new Error("SafeCAudio: call SafeCAudio.attach(instance) after instantiating the module");
        return instance.exports;
    };
    const f32 = (ptr, n) => new Float32Array(exp().memory.buffer, ptr, n);
    const u8 = (ptr, n) => new Uint8Array(exp().memory.buffer, ptr, n);

    // ── audio ─────────────────────────────────────────────────────────────

    const WORKLET_SOURCE = `
class SafeCAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.outCh = o.outCh; this.inCh = o.inCh; this.frames = o.frames; this.nbuf = o.nbuf;
    this.queue = []; this.cur = null; this.pos = 0; this.alive = true;
    this.inBuf = this.inCh ? new Float32Array(this.inCh * this.frames) : null; this.inPos = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d.type === 'out') this.queue.push(d.buf);
      else if (d.type === 'reset') { this.queue = []; this.cur = null; this.inPos = 0; if (this.outCh) this.request(this.nbuf); }
      else if (d.type === 'close') this.alive = false;
    };
    if (this.outCh) this.request(this.nbuf);
  }
  request(n) { for (let i = 0; i < n; i++) this.port.postMessage({ type: 'need' }); }
  process(inputs, outputs) {
    const input = inputs[0] || [];
    const output = outputs[0] || [];
    const n = output.length ? output[0].length : (input.length ? input[0].length : 128);
    if (this.inCh) {
      for (let f = 0; f < n; f++) {
        for (let c = 0; c < this.inCh; c++) this.inBuf[c * this.frames + this.inPos] = input[c] ? input[c][f] : 0;
        if (++this.inPos >= this.frames) {
          this.port.postMessage({ type: 'in', buf: this.inBuf }, [this.inBuf.buffer]);
          this.inBuf = new Float32Array(this.inCh * this.frames);
          this.inPos = 0;
        }
      }
    }
    if (this.outCh) {
      for (let f = 0; f < n; f++) {
        if (!this.cur) {
          this.cur = this.queue.shift() || null;
          this.pos = 0;
          if (!this.cur) { this.port.postMessage({ type: 'xrun' }); break; }
        }
        for (let c = 0; c < output.length; c++)
          output[c][f] = c < this.outCh ? this.cur[c * this.frames + this.pos] : 0;
        if (++this.pos >= this.frames) { this.cur = null; this.port.postMessage({ type: 'need' }); }
      }
    }
    return this.alive;
  }
}
registerProcessor('safec-audio', SafeCAudioProcessor);
`;

    // One period through the module. 'inputs': an array of per-channel
    // Float32Arrays (or one planar Float32Array), null for silence.
    function render(s, inputs, outputs) {
        const e = exp();
        const frames = s.frames;
        if (s.inCh) {
            const view = f32(e.std_audio_web_buffer(s.handle, 1), s.inCh * frames);
            for (let c = 0; c < s.inCh; c++) {
                for (let i = 0; i < frames; i++) {
                    let v = 0;
                    if (inputs instanceof Float32Array) v = inputs[c * frames + i];
                    else if (inputs && inputs[c]) v = inputs[c][i] || 0;
                    view[c * frames + i] = v;
                }
            }
        }
        const r = e.std_audio_web_process(s.handle, frames);
        if (s.outCh) {
            const view = f32(e.std_audio_web_buffer(s.handle, 0), s.outCh * frames);
            if (outputs instanceof Float32Array) outputs.set(view);
            else for (let c = 0; c < outputs.length; c++) {
                const o = outputs[c];
                if (c < s.outCh) o.set(view.subarray(c * frames, (c + 1) * frames));
                else o.fill(0);
            }
        }
        return r;
    }

    function makeContext(rate) {
        const AC = options.AudioContext;
        if (!AC) return null;
        let ctx = null;
        try { ctx = new AC({ sampleRate: rate, latencyHint: "interactive" }); } catch (_) { ctx = null; }
        if (!ctx) { try { ctx = new AC(); } catch (_) { return null; } }
        if (ctx.sampleRate !== rate) { try { ctx.close(); } catch (_) {} return null; }
        return ctx;
    }

    function connect(s) {
        if (!s.node || s.connected) return;
        if (s.inCh && inputStream && s.ctx.createMediaStreamSource) {
            s.source = s.ctx.createMediaStreamSource(inputStream);
            s.source.connect(s.node);
        }
        s.node.connect(s.ctx.destination);
        s.connected = true;
    }

    function disconnect(s) {
        if (!s.connected) return;
        try { if (s.source) s.source.disconnect(); } catch (_) {}
        try { s.node.disconnect(); } catch (_) {}
        s.source = null;
        s.connected = false;
    }

    function openWorklet(s) {
        const blob = typeof Blob !== "undefined" ? new Blob([WORKLET_SOURCE], { type: "application/javascript" }) : null;
        const url = blob && typeof URL !== "undefined" && URL.createObjectURL ? URL.createObjectURL(blob)
                                                                          : "data:application/javascript," + encodeURIComponent(WORKLET_SOURCE);
        s.ready = s.ctx.audioWorklet.addModule(url).then(() => {
            const Node = options.AudioWorkletNode || root.AudioWorkletNode;
            s.node = new Node(s.ctx, "safec-audio", {
                numberOfInputs: s.inCh ? 1 : 0,
                numberOfOutputs: s.outCh ? 1 : 0,
                outputChannelCount: s.outCh ? [s.outCh] : [],
                processorOptions: { outCh: s.outCh, inCh: s.inCh, frames: s.frames, nbuf: s.nbuf },
            });
            s.node.port.onmessage = (ev) => {
                const d = ev.data;
                const e = exp();
                if (d.type === "need") {
                    let input = null;
                    if (s.inCh) {
                        input = s.inQueue.shift() || null;
                        if (!input && s.inSeen) e.std_audio_web_xrun(s.handle, 1);
                    }
                    const buf = new Float32Array(s.outCh * s.frames);
                    render(s, input, buf);
                    s.node.port.postMessage({ type: "out", buf }, [buf.buffer]);
                } else if (d.type === "in") {
                    s.inSeen = true;
                    if (!s.outCh) { render(s, d.buf, null); return; }
                    s.inQueue.push(d.buf);
                    if (s.inQueue.length > s.nbuf + 2) { s.inQueue.shift(); e.std_audio_web_xrun(s.handle, 1); }
                } else if (d.type === "xrun") {
                    if (s.running) e.std_audio_web_xrun(s.handle, 2);
                }
            };
            if (s.running) connect(s);
        });
    }

    function openScript(s) {
        const ins = s.inCh ? s.inCh : 1;
        const outs = s.outCh ? s.outCh : 1;
        s.node = s.ctx.createScriptProcessor(s.frames, ins, outs);
        s.node.onaudioprocess = (ev) => {
            const inputs = [];
            for (let c = 0; c < s.inCh; c++) inputs.push(ev.inputBuffer.getChannelData(c));
            const outputs = [];
            for (let c = 0; c < ev.outputBuffer.numberOfChannels; c++) outputs.push(ev.outputBuffer.getChannelData(c));
            if (!s.running) { outputs.forEach((o) => o.fill(0)); return; }
            render(s, inputs, outputs);
        };
        s.ready = Promise.resolve();
    }

    const audioImports = {
        std_audio_js_now_ms: () => options.now(),
        std_audio_web_available: () => (options.AudioContext ? 1 : 0),
        std_audio_web_default_rate: () => {
            if (options.defaultRate) return options.defaultRate;
            const AC = options.AudioContext;
            if (!AC) return 48000;
            try { const c = new AC(); options.defaultRate = c.sampleRate | 0; if (c.close) c.close(); } catch (_) { options.defaultRate = 48000; }
            return options.defaultRate;
        },
        std_audio_web_output_channels: () => 2,
        std_audio_web_input_channels: () => inputChannels,
        std_audio_web_open: (handle, outCh, inCh, rate, frames, nbuf) => {
            const ctx = makeContext(rate);
            if (!ctx) return -1;
            const s = { handle, ctx, outCh, inCh, frames, nbuf, node: null, source: null, connected: false,
                        running: false, inQueue: [], inSeen: false, ready: null, mode: "" };
            if (options.worklet && ctx.audioWorklet) {
                s.mode = "worklet";
                openWorklet(s);
            } else if (ctx.createScriptProcessor) {
                // ScriptProcessor periods are a power of two, 256 .. 16384.
                let n = 256;
                while (n < frames && n < 16384) n *= 2;
                s.frames = n;
                s.mode = "script";
                openScript(s);
            } else {
                try { ctx.close(); } catch (_) {}
                return -2;
            }
            streams[handle] = s;
            return s.frames;
        },
        std_audio_web_latency: (handle) => {
            const s = streams[handle];
            if (!s) return 0;
            const base = Math.round(((s.ctx.baseLatency || 0) + (s.ctx.outputLatency || 0)) * s.ctx.sampleRate);
            return (s.mode === "worklet" ? s.frames * s.nbuf + 128 : s.frames * 2) + base;
        },
        std_audio_web_start: (handle) => {
            const s = streams[handle];
            if (!s) return -1;
            s.running = true;
            s.inQueue = [];
            s.inSeen = false;
            if (s.node && s.mode === "worklet") s.node.port.postMessage({ type: "reset" });
            connect(s);
            if (s.ctx.resume) { const p = s.ctx.resume(); if (p && p.catch) p.catch(() => {}); }
            return 0;
        },
        std_audio_web_stop: (handle) => {
            const s = streams[handle];
            if (!s) return;
            s.running = false;
            disconnect(s);
            if (s.ctx.suspend) { const p = s.ctx.suspend(); if (p && p.catch) p.catch(() => {}); }
        },
        std_audio_web_close: (handle) => {
            const s = streams[handle];
            if (!s) return;
            s.running = false;
            disconnect(s);
            if (s.node && s.mode === "worklet") s.node.port.postMessage({ type: "close" });
            if (s.ctx.close) { const p = s.ctx.close(); if (p && p.catch) p.catch(() => {}); }
            streams[handle] = undefined;
        },
    };

    // ── MIDI ──────────────────────────────────────────────────────────────

    function refreshPorts() {
        if (!midi.access) return;
        midi.inputs = Array.from(midi.access.inputs.values());
        midi.outputs = Array.from(midi.access.outputs.values());
    }

    function writeString(ptr, cap, text) {
        const bytes = new TextEncoder().encode(text);
        const n = Math.min(bytes.length, cap - 1);
        if (n < 0) return -1;
        const view = u8(ptr, cap);
        view.set(bytes.subarray(0, n));
        view[n] = 0;
        return n;
    }

    const midiImports = {
        std_midi_web_available: () => (midi.access ? 1 : 0),
        std_midi_web_port_count: (input) => {
            refreshPorts();
            return (input ? midi.inputs : midi.outputs).length;
        },
        std_midi_web_port_name: (input, index, ptr, cap) => {
            refreshPorts();
            const p = (input ? midi.inputs : midi.outputs)[index];
            if (!p) return -1;
            const name = p.name + (p.manufacturer ? " (" + p.manufacturer + ")" : "");
            return writeString(ptr, cap, name);
        },
        std_midi_web_open: (handle, input, index) => {
            refreshPorts();
            const p = (input ? midi.inputs : midi.outputs)[index];
            if (!p) return -1;
            midi.open[handle] = { port: p, input: !!input };
            if (input) {
                p.onmidimessage = (ev) => {
                    const data = ev.data;
                    const e = exp();
                    const ptr = e.std_midi_web_scratch(handle, data.length);
                    if (!ptr) return;
                    u8(ptr, data.length).set(data);
                    e.std_midi_web_receive(handle, data.length, ev.timeStamp || options.now());
                };
            }
            if (p.open) { const r = p.open(); if (r && r.catch) r.catch(() => {}); }
            return 0;
        },
        std_midi_web_send: (handle, ptr, len, ts) => {
            const o = midi.open[handle];
            if (!o || o.input) return -1;
            try { o.port.send(Array.from(u8(ptr, len))); } catch (_) { return -1; }
            return 0;
        },
        std_midi_web_close: (handle) => {
            const o = midi.open[handle];
            if (!o) return;
            if (o.input) o.port.onmidimessage = null;
            midi.open[handle] = undefined;
        },
    };

    const SafeCAudio = {
        options,
        // The import functions std/audio declares, merged into 'env'.
        imports(env) { return Object.assign(env || {}, audioImports, midiImports); },
        attach(inst) { instance = inst; return inst; },
        // Ask for Web MIDI (a permission prompt in some browsers).
        async enableMIDI(opts) {
            const nav = options.navigator;
            if (!nav || !nav.requestMIDIAccess) throw new Error("SafeCAudio: Web MIDI is not available here");
            midi.access = await nav.requestMIDIAccess({ sysex: !!(opts && opts.sysex) });
            refreshPorts();
            midi.access.onstatechange = refreshPorts;
            return midi.access;
        },
        // Make the microphone an input device (call before Audio::open()).
        async enableInput(constraints) {
            const nav = options.navigator;
            if (!nav || !nav.mediaDevices || !nav.mediaDevices.getUserMedia) throw new Error("SafeCAudio: no getUserMedia here");
            inputStream = await nav.mediaDevices.getUserMedia(constraints || { audio: true });
            const track = inputStream.getAudioTracks()[0];
            const st = track && track.getSettings ? track.getSettings() : {};
            inputChannels = st.channelCount || 1;
            return inputStream;
        },
        // Resume every stream's context (call from a user gesture).
        resume() {
            for (const s of streams) if (s && s.ctx.resume) { const p = s.ctx.resume(); if (p && p.catch) p.catch(() => {}); }
        },
        // For tests: a stream's state ({ mode, frames, ctx, node, ... }).
        _stream(handle) { return streams[handle]; },
        WORKLET_SOURCE,
    };

    root.SafeCAudio = SafeCAudio;
    if (typeof module !== "undefined" && module.exports) module.exports = SafeCAudio;
})(typeof globalThis !== "undefined" ? globalThis : this);
