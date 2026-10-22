// Drives audio_web_check.wasm through std/audio/audio_web.js under Node,
// with stand-ins for the browser APIs it uses: AudioContext with a
// ScriptProcessorNode, AudioWorklet (the glue's processor source is run
// here, with real MessagePorts between it and the main thread), and Web
// MIDI. usage: node run.mjs <module.wasm> <path to audio_web.js>
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const [wasmPath, gluePath] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const SafeCAudio = require(gluePath);
let failures = 0;
const check = (ok, what) => { console.log(`  ${ok ? "PASS" : "FAIL"}  ${what}`); if (!ok) failures++; };
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── fakes ───────────────────────────────────────────────────────────────
class FakeNode {
    constructor() { this.connections = []; }
    connect(n) { this.connections.push(n); }
    disconnect() { this.connections = []; }
}
class FakeScriptProcessor extends FakeNode {
    constructor(n, ins, outs) { super(); this.bufferSize = n; this.ins = ins; this.outs = outs; this.onaudioprocess = null; }
    fire() {
        const mk = (c) => { const chans = []; for (let i = 0; i < c; i++) chans.push(new Float32Array(this.bufferSize));
                            return { numberOfChannels: c, getChannelData: (i) => chans[i] }; };
        const ev = { inputBuffer: mk(this.ins), outputBuffer: mk(this.outs) };
        this.onaudioprocess(ev);
        return ev.outputBuffer;
    }
}
let registered = null;
class FakeContext {
    constructor(opts) {
        this.sampleRate = (opts && opts.sampleRate) || 48000;
        this.destination = new FakeNode();
        this.state = "suspended";
        this.baseLatency = 0;
        FakeContext.last = this;
        if (FakeContext.worklet) {
            this.audioWorklet = {
                addModule: async () => {
                    const define = new Function("AudioWorkletProcessor", "registerProcessor", SafeCAudio.WORKLET_SOURCE);
                    class AudioWorkletProcessor { constructor() { this.port = FakeContext.processorPort; } }
                    define(AudioWorkletProcessor, (name, cls) => { registered = cls; });
                },
            };
        }
    }
    createScriptProcessor(n, ins, outs) { this.node = new FakeScriptProcessor(n, ins, outs); return this.node; }
    createMediaStreamSource() { return new FakeNode(); }
    resume() { this.state = "running"; return Promise.resolve(); }
    suspend() { this.state = "suspended"; return Promise.resolve(); }
    close() { this.state = "closed"; return Promise.resolve(); }
}
class FakeWorkletNode extends FakeNode {
    constructor(ctx, name, opts) {
        super();
        const ch = new MessageChannel();
        this.port = ch.port1;
        FakeContext.processorPort = ch.port2;
        this.processor = new registered({ processorOptions: opts.processorOptions });
        FakeWorkletNode.last = this;
    }
}

// ── the module ──────────────────────────────────────────────────────────
SafeCAudio.options.AudioContext = FakeContext;
SafeCAudio.options.AudioWorkletNode = FakeWorkletNode;
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), { env: SafeCAudio.imports({}) });
SafeCAudio.attach(instance);
const T = instance.exports;

// ScriptProcessor: the callback runs inside onaudioprocess.
SafeCAudio.options.worklet = false;
FakeContext.worklet = false;
{
    const frames = T.test_open(300);
    check(frames === 512, `ScriptProcessor: period rounded to a power of two (got ${frames})`);
    const node = FakeContext.last.node;
    check(node && node.connections.length === 1 && FakeContext.last.state === "running", "ScriptProcessor: connected and resumed on start()");
    let ok = true;
    for (let k = 0; k < 4; k++) {
        const out = node.fire();
        const l = out.getChannelData(0), r = out.getChannelData(1);
        for (let i = 0; i < 512; i++) if (l[i] !== i + 1 || r[i] !== -(i + 1)) ok = false;
    }
    check(ok, "ScriptProcessor: callback output reaches both channels, deinterleaved");
    check(T.test_calls() === 4, `ScriptProcessor: one callback per onaudioprocess (got ${T.test_calls()})`);
    check(T.test_close() === 0 && FakeContext.last.state === "closed", "ScriptProcessor: stop() and close()");
}

// AudioWorklet: the worklet asks for periods, plays them from its queue.
SafeCAudio.options.worklet = true;
FakeContext.worklet = true;
{
    const frames = T.test_open(256);
    check(frames === 256, `AudioWorklet: requested period kept (got ${frames})`);
    for (let i = 0; i < 20 && !FakeWorkletNode.last; i++) await tick();
    const node = FakeWorkletNode.last;
    check(!!node, "AudioWorklet: module loaded, node created");
    const proc = node.processor;
    for (let i = 0; i < 20; i++) await tick();      // 'need' x3 -> rendered periods queued
    check(proc.queue.length === 3 && T.test_calls() === 3, `AudioWorklet: queue primed with buffer_count periods (${proc.queue.length}, ${T.test_calls()} calls)`);
    const played = [];
    for (let q = 0; q < 8; q++) {                  // 8 render quanta of 128 = 4 periods
        const out = [new Float32Array(128), new Float32Array(128)];
        proc.process([], [out]);
        played.push(out);
        for (let i = 0; i < 5; i++) await tick();
    }
    let ok = true;
    for (let q = 0; q < 8; q++) for (let i = 0; i < 128; i++) {
        const want = ((q % 2) * 128) + i + 1;
        if (played[q][0][i] !== want || played[q][1][i] !== -want) ok = false;
    }
    check(ok, "AudioWorklet: quanta play the periods in order");
    check(T.test_calls() === 7, `AudioWorklet: a new period for each one played (${T.test_calls()} calls)`);
    // Starve it (play faster than the main thread answers): the worklet
    // reports an underrun, a following callback sees it.
    for (let q = 0; q < 10; q++) proc.process([], [[new Float32Array(128), new Float32Array(128)]]);
    for (let i = 0; i < 20; i++) await tick();
    for (let q = 0; q < 4; q++) { proc.process([], [[new Float32Array(128), new Float32Array(128)]]); for (let i = 0; i < 5; i++) await tick(); }
    check((T.test_status() & 2) !== 0, "AudioWorklet: a dry queue arrives as AUDIO_STATUS_OUTPUT_UNDERFLOW");
    check(T.test_close() === 0, "AudioWorklet: stop() and close()");
}

// Input through the worklet: microphone frames reach an i16 callback.
{
    SafeCAudio.options.navigator = {
        mediaDevices: { getUserMedia: async () => ({ getAudioTracks: () => [{ getSettings: () => ({ channelCount: 1 }) }] }) },
    };
    await SafeCAudio.enableInput();
    FakeWorkletNode.last = null;
    const frames = T.test_open_input();
    check(frames === 128, `AudioWorklet input: stream opened (got ${frames})`);
    for (let i = 0; i < 20 && !FakeWorkletNode.last; i++) await tick();
    const proc = FakeWorkletNode.last.processor;
    check(FakeWorkletNode.last.connections.length === 0, "AudioWorklet input: an output-less node is not connected to the destination");
    for (let q = 0; q < 4; q++) {
        const mic = new Float32Array(128).fill(0.5);
        proc.process([[mic]], []);
        for (let i = 0; i < 5; i++) await tick();
    }
    // 4 periods of 128 samples at 0.5 -> 16384 each in i16.
    check(T.test_in_calls() === 4, `AudioWorklet input: one callback per period (${T.test_in_calls()})`);
    check(Math.abs(T.test_in_sum() - 4 * 128 * 16384) < 1, `AudioWorklet input: samples converted to i16 (${T.test_in_sum()})`);
    check(T.test_close() === 0, "AudioWorklet input: stop() and close()");
}

// Web MIDI.
{
    const sent = [];
    const input = { name: "Fake In", manufacturer: "SafeC", onmidimessage: null, open: () => Promise.resolve() };
    const output = { name: "Fake Out", manufacturer: "", send: (d) => sent.push(d), open: () => Promise.resolve() };
    SafeCAudio.options.navigator = {
        requestMIDIAccess: async () => ({ inputs: new Map([["i", input]]), outputs: new Map([["o", output]]) }),
    };
    await SafeCAudio.enableMIDI({ sysex: true });
    check(T.test_midi_open() === 0, "Web MIDI: ports listed and opened; no virtual ports");
    check(sent.length === 1 && sent[0].join(",") === "147,64,99", `Web MIDI: send() (got ${JSON.stringify(sent)})`);
    input.onmidimessage({ data: new Uint8Array([0xB0, 7, 100]), timeStamp: 12.5 });
    input.onmidimessage({ data: new Uint8Array([0xF8]), timeStamp: 13 });      // clock: filtered by default
    check(T.test_midi_poll() === ((0xB0 << 16) | (7 << 8) | 100), "Web MIDI: an incoming message reaches the queue");
    check(T.test_midi_poll() === 0, "Web MIDI: timing messages ignored by default");
    check(T.test_midi_close() === 0 && input.onmidimessage === null, "Web MIDI: close() detaches the port");
}

console.log(failures === 0 ? "audio web: all passed" : `audio web: ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
