// SafeC Standard Library — reinforcement learning (see rl.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/rl.h>
#include <std/ml/module.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/math.h>

namespace std {

// Persistent tensors (environment and agent state) live on the heap, also
// when the caller has an arena active.
static ?&TensorArena rl_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void rl_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { a.begin(); }
        case null: { }
    }
}

static void rl_drop_(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

// Argmax with ties broken at random.
static unsigned long rl_argmax_(const float* v, unsigned long n, &Rng rng) {
    unsigned long best = 0UL;
    unsigned long ties = 1UL;
    unsafe {
        unsigned long i = 1UL;
        while (i < n) {
            if (v[i] > v[best]) { best = i; ties = 1UL; }
            else if (v[i] == v[best]) {
                ties = ties + 1UL;
                if (rng.below(ties) == 0UL) best = i;
            }
            i = i + 1UL;
        }
    }
    return best;
}

// obs (any shape, obsDim values) as a [1, obsDim] row.
static &Tensor rl_row_(const &Tensor obs, unsigned long dim) {
    &Tensor r = tensor_new_2d(1UL, dim, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < dim) { r->data[i] = obs->data[i]; i = i + 1UL; }
    }
    return r;
}

// ══ Bandit ═══════════════════════════════════════════════════════════════════

struct Bandit bandit_new(unsigned long k, unsigned long seed) {
    struct Bandit b;
    b.k = k;
    b.rng = rng_new(seed);
    ?&TensorArena cur = rl_heap_begin_();
    b.means = tensor_new_1d(k, 0);
    rl_heap_end_(cur);
    b.rng.fill_normal(b.means, (float)0.0, (float)1.0);
    return b;
}

float Bandit::pull(unsigned long arm) {
    unsafe { return self.means->data[arm] + self.rng.normal(); }
}

unsigned long Bandit::best_arm() const {
    unsigned long best = 0UL;
    unsafe {
        unsigned long i = 1UL;
        while (i < self.k) { if (self.means->data[i] > self.means->data[best]) best = i; i = i + 1UL; }
    }
    return best;
}

void Bandit::free() { rl_drop_(self.means); }

// ══ GridWorld ════════════════════════════════════════════════════════════════

struct GridWorld gridworld_new(unsigned long width, unsigned long height, unsigned long seed) {
    struct GridWorld g;
    g.width = width;
    g.height = height;
    ?&TensorArena cur = rl_heap_begin_();
    g.cells = tensor_new_1d(width * height, 0);
    rl_heap_end_(cur);
    g.start = 0UL;
    g.state = 0UL;
    g.stepReward = (float)-1.0;
    g.pitReward = (float)-100.0;
    g.goalReward = (float)0.0;
    g.pitTerminal = 0;
    g.slip = (float)0.0;
    g.steps = 0UL;
    g.maxSteps = 1000UL;
    g.rng = rng_new(seed);
    g.set(width - 1UL, height - 1UL, GRID_GOAL);
    return g;
}

struct GridWorld gridworld_cliff(unsigned long seed) {
    struct GridWorld g = gridworld_new(12UL, 4UL, seed);
    g.set_start(0UL, 3UL);
    unsigned long x = 1UL;
    while (x < 11UL) { g.set(x, 3UL, GRID_PIT); x = x + 1UL; }
    return g;
}

void GridWorld::set(unsigned long x, unsigned long y, int cell) {
    unsafe { self.cells->data[y * self.width + x] = (float)cell; }
}

void GridWorld::set_start(unsigned long x, unsigned long y) {
    self.start = y * self.width + x;
}

unsigned long GridWorld::states() const { return self.width * self.height; }

unsigned long GridWorld::reset() {
    self.state = self.start;
    self.steps = 0UL;
    return self.state;
}

static int grid_cell_(const &GridWorld g, unsigned long s) {
    unsafe { return (int)g.cells->data[s]; }
}

struct EnvStep GridWorld::step(unsigned long action, &unsigned long next) {
    struct EnvStep r;
    r.reward = self.stepReward;
    r.terminated = 0;
    r.truncated = 0;
    if (self.slip > (float)0.0 && self.rng.uniform() < self.slip) action = self.rng.below(4UL);
    unsigned long x = self.state % self.width;
    unsigned long y = self.state / self.width;
    if (action == GRID_UP && y > 0UL) y = y - 1UL;
    else if (action == GRID_RIGHT && x + 1UL < self.width) x = x + 1UL;
    else if (action == GRID_DOWN && y + 1UL < self.height) y = y + 1UL;
    else if (action == GRID_LEFT && x > 0UL) x = x - 1UL;
    unsigned long s = y * self.width + x;
    int cell = grid_cell_(self, s);
    if (cell != GRID_WALL) self.state = s;
    if (cell == GRID_PIT) {
        r.reward = self.pitReward;
        if (self.pitTerminal) r.terminated = 1;
        else self.state = self.start;
    } else if (cell == GRID_GOAL) {
        r.reward = r.reward + self.goalReward;
        r.terminated = 1;
    }
    self.steps = self.steps + 1UL;
    if (!r.terminated && self.steps >= self.maxSteps) r.truncated = 1;
    *next = self.state;
    return r;
}

void GridWorld::one_hot(unsigned long state, &Tensor obs) const {
    unsafe {
        unsigned long n = self.width * self.height;
        unsigned long i = 0UL;
        while (i < n) { obs->data[i] = i == state ? (float)1.0 : (float)0.0; i = i + 1UL; }
    }
}

void GridWorld::free() { rl_drop_(self.cells); }

// ══ CartPole ═════════════════════════════════════════════════════════════════

static void cartpole_obs_(const &CartPole env, &Tensor obs) {
    unsafe {
        obs->data[0] = env.x;
        obs->data[1] = env.xDot;
        obs->data[2] = env.theta;
        obs->data[3] = env.thetaDot;
    }
}

struct CartPole cartpole_new(unsigned long seed) {
    struct CartPole env;
    env.x = (float)0.0; env.xDot = (float)0.0; env.theta = (float)0.0; env.thetaDot = (float)0.0;
    env.steps = 0UL;
    env.maxSteps = 500UL;
    env.rng = rng_new(seed);
    return env;
}

void CartPole::reset(&Tensor obs) {
    self.x = (float)0.1 * self.rng.uniform() - (float)0.05;
    self.xDot = (float)0.1 * self.rng.uniform() - (float)0.05;
    self.theta = (float)0.1 * self.rng.uniform() - (float)0.05;
    self.thetaDot = (float)0.1 * self.rng.uniform() - (float)0.05;
    self.steps = 0UL;
    cartpole_obs_(self, obs);
}

struct EnvStep CartPole::step(unsigned long action, &Tensor obs) {
    float gravity = (float)9.8;
    float massPole = (float)0.1;
    float totalMass = (float)1.1;
    float length = (float)0.5;            // half the pole
    float poleMassLength = (float)0.05;
    float tau = (float)0.02;
    float force = action == 1UL ? (float)10.0 : (float)-10.0;
    float c = cos_f(self.theta);
    float s = sin_f(self.theta);
    float temp = (force + poleMassLength * self.thetaDot * self.thetaDot * s) / totalMass;
    float thetaAcc = (gravity * s - c * temp) /
                     (length * ((float)4.0 / (float)3.0 - massPole * c * c / totalMass));
    float xAcc = temp - poleMassLength * thetaAcc * c / totalMass;
    self.x = self.x + tau * self.xDot;
    self.xDot = self.xDot + tau * xAcc;
    self.theta = self.theta + tau * self.thetaDot;
    self.thetaDot = self.thetaDot + tau * thetaAcc;
    self.steps = self.steps + 1UL;
    cartpole_obs_(self, obs);
    struct EnvStep r;
    r.reward = (float)1.0;
    float lim = (float)0.20943951;        // 12 degrees
    r.terminated = self.x < (float)-2.4 || self.x > (float)2.4 || self.theta < -lim || self.theta > lim;
    r.truncated = !r.terminated && self.steps >= self.maxSteps;
    return r;
}

// ══ Bandit agents ════════════════════════════════════════════════════════════

struct BanditAgent bandit_agent_new(int kind, unsigned long k, float param, unsigned long seed) {
    struct BanditAgent a;
    a.kind = kind;
    a.k = k;
    ?&TensorArena cur = rl_heap_begin_();
    a.counts = tensor_new_1d(k, 0);
    a.values = tensor_new_1d(k, 0);
    rl_heap_end_(cur);
    a.param = param;
    a.baseline = (float)0.0;
    a.t = 0UL;
    a.rng = rng_new(seed);
    return a;
}

static void bandit_softmax_(const &BanditAgent a, float* p) {
    unsafe {
        float mx = a.values->data[0];
        unsigned long i = 1UL;
        while (i < a.k) { if (a.values->data[i] > mx) mx = a.values->data[i]; i = i + 1UL; }
        float s = (float)0.0;
        i = 0UL;
        while (i < a.k) { p[i] = exp_f(a.values->data[i] - mx); s = s + p[i]; i = i + 1UL; }
        i = 0UL;
        while (i < a.k) { p[i] = p[i] / s; i = i + 1UL; }
    }
}

unsigned long BanditAgent::select() {
    unsigned long k = self.k;
    unsigned long choice = 0UL;
    unsafe {
        float* score = (float*)malloc(sizeof(float) * k);
        const float* n = (const float*)self.counts->data;
        const float* q = (const float*)self.values->data;
        unsigned long i = 0UL;
        if (self.kind == BANDIT_EPS_GREEDY) {
            if (self.rng.uniform() < self.param) choice = self.rng.below(k);
            else choice = rl_argmax_(q, k, &self.rng);
        } else if (self.kind == BANDIT_UCB1) {
            int untried = 0;
            while (i < k) { if (n[i] == (float)0.0) { untried = 1; choice = i; break; } i = i + 1UL; }
            if (!untried) {
                float lt = log_f((float)self.t);
                i = 0UL;
                while (i < k) { score[i] = q[i] + self.param * sqrt_f(lt / n[i]); i = i + 1UL; }
                choice = rl_argmax_((const float*)score, k, &self.rng);
            }
        } else if (self.kind == BANDIT_THOMPSON) {
            // posterior of each arm's mean under a N(0, param^2) prior and unit noise
            float prior = (float)1.0 / (self.param * self.param);
            while (i < k) {
                float prec = prior + n[i];
                score[i] = n[i] * q[i] / prec + self.rng.normal() / sqrt_f(prec);
                i = i + 1UL;
            }
            choice = rl_argmax_((const float*)score, k, &self.rng);
        } else {
            bandit_softmax_(self, score);
            float u = self.rng.uniform();
            float acc = (float)0.0;
            choice = k - 1UL;
            while (i < k) { acc = acc + score[i]; if (u < acc) { choice = i; break; } i = i + 1UL; }
        }
        free((void*)score);
    }
    return choice;
}

void BanditAgent::update(unsigned long arm, float reward) {
    self.t = self.t + 1UL;
    unsafe {
        self.counts->data[arm] = self.counts->data[arm] + (float)1.0;
        if (self.kind == BANDIT_GRADIENT) {
            float* p = (float*)malloc(sizeof(float) * self.k);
            bandit_softmax_(self, p);
            float d = reward - self.baseline;
            unsigned long i = 0UL;
            while (i < self.k) {
                float ind = i == arm ? (float)1.0 : (float)0.0;
                self.values->data[i] = self.values->data[i] + self.param * d * (ind - p[i]);
                i = i + 1UL;
            }
            free((void*)p);
        } else {
            float n = self.counts->data[arm];
            self.values->data[arm] = self.values->data[arm] + (reward - self.values->data[arm]) / n;
        }
    }
    self.baseline = self.baseline + (reward - self.baseline) / (float)self.t;
}

void BanditAgent::free() {
    rl_drop_(self.counts);
    rl_drop_(self.values);
}

// ══ Tabular TD ═══════════════════════════════════════════════════════════════

struct TabularAgent tabular_agent_new(unsigned long states, unsigned long actions,
                                      float alpha, float gamma, float epsilon, unsigned long seed) {
    struct TabularAgent a;
    ?&TensorArena cur = rl_heap_begin_();
    a.q = tensor_new_2d(states, actions, 0);
    rl_heap_end_(cur);
    a.states = states;
    a.actions = actions;
    a.alpha = alpha;
    a.gamma = gamma;
    a.epsilon = epsilon;
    a.rng = rng_new(seed);
    return a;
}

static float tab_q_(const &TabularAgent a, unsigned long s, unsigned long act) {
    unsafe { return a.q->data[s * a.actions + act]; }
}

static float tab_max_(const &TabularAgent a, unsigned long s) {
    float m = tab_q_(a, s, 0UL);
    unsigned long i = 1UL;
    while (i < a.actions) { float v = tab_q_(a, s, i); if (v > m) m = v; i = i + 1UL; }
    return m;
}

unsigned long TabularAgent::greedy(unsigned long state) const {
    unsigned long best = 0UL;
    unsigned long i = 1UL;
    while (i < self.actions) { if (tab_q_(self, state, i) > tab_q_(self, state, best)) best = i; i = i + 1UL; }
    return best;
}

unsigned long TabularAgent::act(unsigned long state) {
    if (self.rng.uniform() < self.epsilon) return self.rng.below(self.actions);
    unsafe { return rl_argmax_((const float*)&self.q->data[state * self.actions], self.actions, &self.rng); }
}

static float tab_apply_(&TabularAgent a, unsigned long s, unsigned long act, float target) {
    float td = target - tab_q_(a, s, act);
    unsafe { a.q->data[s * a.actions + act] = a.q->data[s * a.actions + act] + a.alpha * td; }
    return td;
}

float TabularAgent::q_learning_update(unsigned long s, unsigned long act, float r, unsigned long s2, int terminal) {
    float target = terminal ? r : r + self.gamma * tab_max_(self, s2);
    return tab_apply_(self, s, act, target);
}

float TabularAgent::sarsa_update(unsigned long s, unsigned long act, float r, unsigned long s2, unsigned long a2,
                                 int terminal) {
    float target = terminal ? r : r + self.gamma * tab_q_(self, s2, a2);
    return tab_apply_(self, s, act, target);
}

float TabularAgent::expected_sarsa_update(unsigned long s, unsigned long act, float r, unsigned long s2, int terminal) {
    float target = r;
    if (!terminal) {
        float mean = (float)0.0;
        unsigned long i = 0UL;
        while (i < self.actions) { mean = mean + tab_q_(self, s2, i); i = i + 1UL; }
        mean = mean / (float)self.actions;
        float ev = ((float)1.0 - self.epsilon) * tab_max_(self, s2) + self.epsilon * mean;
        target = r + self.gamma * ev;
    }
    return tab_apply_(self, s, act, target);
}

void TabularAgent::free() { rl_drop_(self.q); }

// ══ DQN ══════════════════════════════════════════════════════════════════════

struct ReplayBuffer replay_new(unsigned long slots, unsigned long obsDim) {
    struct ReplayBuffer rb;
    ?&TensorArena cur = rl_heap_begin_();
    rb.obs = tensor_new_2d(slots, obsDim, 0);
    rb.next = tensor_new_2d(slots, obsDim, 0);
    rb.action = tensor_new_1d(slots, 0);
    rb.reward = tensor_new_1d(slots, 0);
    rb.terminal = tensor_new_1d(slots, 0);
    rl_heap_end_(cur);
    rb.slots = slots;
    rb.size = 0UL;
    rb.pos = 0UL;
    rb.obsDim = obsDim;
    return rb;
}

void ReplayBuffer::push(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    unsigned long p = self.pos;
    unsigned long d = self.obsDim;
    unsafe {
        unsigned long i = 0UL;
        while (i < d) {
            self.obs->data[p * d + i] = obs->data[i];
            self.next->data[p * d + i] = next->data[i];
            i = i + 1UL;
        }
        self.action->data[p] = (float)action;
        self.reward->data[p] = reward;
        self.terminal->data[p] = terminal ? (float)1.0 : (float)0.0;
    }
    self.pos = (p + 1UL) % self.slots;
    if (self.size < self.slots) self.size = self.size + 1UL;
}

void ReplayBuffer::free() {
    rl_drop_(self.obs); rl_drop_(self.next); rl_drop_(self.action); rl_drop_(self.reward); rl_drop_(self.terminal);
}

struct DqnConfig dqn_config_default() {
    struct DqnConfig c;
    c.hidden = 64UL;
    c.hiddenLayers = 2UL;
    c.gamma = (float)0.99;
    c.lr = (float)0.0005;
    c.batchSize = 64UL;
    c.bufferSize = 50000UL;
    c.learningStarts = 1000UL;
    c.trainEvery = 1UL;
    c.targetEvery = 0UL;
    c.tau = (float)0.005;
    c.epsStart = (float)1.0;
    c.epsEnd = (float)0.05;
    c.epsDecaySteps = 10000UL;
    c.doubleDqn = 1;
    c.huberDelta = (float)1.0;
    c.clipNorm = (float)10.0;
    c.seed = 1UL;
    return c;
}

struct Dqn dqn_new(unsigned long obsDim, unsigned long actions, const &DqnConfig cfg) {
    struct Dqn d;
    d.cfg = *cfg;
    d.rng = rng_new(cfg.seed);
    d.q = mlp_new(obsDim, cfg.hidden, cfg.hiddenLayers, actions, ACT_RELU, ACT_NONE, &d.rng);
    d.target = mlp_new(obsDim, cfg.hidden, cfg.hiddenLayers, actions, ACT_RELU, ACT_NONE, &d.rng);
    model_copy(&d.target, &d.q);
    d.opt = optimizer_new(OPT_ADAM, cfg.lr);
    model_add_params(&d.q, &d.opt);
    d.buffer = replay_new(cfg.bufferSize, obsDim);
    d.stepArena = tensor_arena_new(0UL);
    d.obsDim = obsDim;
    d.actions = actions;
    d.steps = 0UL;
    d.updates = 0UL;
    d.lastLoss = (float)0.0;
    return d;
}

float Dqn::epsilon() const {
    if (self.steps >= self.cfg.epsDecaySteps) return self.cfg.epsEnd;
    float f = (float)self.steps / (float)self.cfg.epsDecaySteps;
    return self.cfg.epsStart + f * (self.cfg.epsEnd - self.cfg.epsStart);
}

unsigned long Dqn::act(const &Tensor obs, int explore) {
    if (explore && self.rng.uniform() < self.epsilon()) return self.rng.below(self.actions);
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    self.stepArena.begin();
    &Tensor q = self.q.forward(rl_row_(obs, self.obsDim));
    unsigned long a;
    unsafe { a = rl_argmax_((const float*)q->data, self.actions, &self.rng); }
    tensor_arena_end();
    self.stepArena.reset();
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    return a;
}

static float dqn_train_(&Dqn d) {
    unsigned long B = d.cfg.batchSize;
    unsigned long D = d.obsDim;
    unsigned long A = d.actions;
    &Tensor ob = tensor_new_2d(B, D, 0);
    &Tensor nx = tensor_new_2d(B, D, 0);
    &Tensor act = tensor_new_1d(B, 0);
    &Tensor y = tensor_new_2d(B, 1UL, 0);
    &Tensor term = tensor_new_1d(B, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < B) {
            unsigned long j = d.rng.below(d.buffer.size);
            unsigned long c = 0UL;
            while (c < D) {
                ob->data[r * D + c] = d.buffer.obs->data[j * D + c];
                nx->data[r * D + c] = d.buffer.next->data[j * D + c];
                c = c + 1UL;
            }
            act->data[r] = d.buffer.action->data[j];
            y->data[r] = d.buffer.reward->data[j];   // the reward; the target is completed below
            term->data[r] = d.buffer.terminal->data[j];
            r = r + 1UL;
        }
    }
    // targets without gradient: r + gamma * (1 - terminal) * Q_target(s', a*)
    tensor_set_grad_enabled(0);
    &Tensor qt = d.target.forward(nx);
    &Tensor qo = d.cfg.doubleDqn ? d.q.forward(nx) : qt;
    unsafe {
        unsigned long r = 0UL;
        while (r < B) {
            float rew = y->data[r];
            unsigned long best = 0UL;
            unsigned long c = 1UL;
            while (c < A) { if (qo->data[r * A + c] > qo->data[r * A + best]) best = c; c = c + 1UL; }
            y->data[r] = term->data[r] > (float)0.5 ? rew : rew + d.cfg.gamma * qt->data[r * A + best];
            r = r + 1UL;
        }
    }
    tensor_set_grad_enabled(1);
    &Tensor qsa = tensor_sum_rows(tensor_mul(d.q.forward(ob), tensor_one_hot(act, A)));
    &Tensor loss = tensor_huber_loss(qsa, y, d.cfg.huberDelta);
    d.opt.zero_grad();
    tensor_backward(loss);
    if (d.cfg.clipNorm > (float)0.0) d.opt.clip_grad_norm(d.cfg.clipNorm);
    d.opt.step();
    d.updates = d.updates + 1UL;
    if (d.cfg.targetEvery > 0UL) {
        if (d.updates % d.cfg.targetEvery == 0UL) model_copy(&d.target, &d.q);
    } else {
        model_soft_update(&d.target, &d.q, d.cfg.tau);
    }
    unsafe { return loss->data[0]; }
}

float Dqn::observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal) {
    self.buffer.push(obs, action, reward, next, terminal);
    self.steps = self.steps + 1UL;
    if (self.buffer.size < self.cfg.learningStarts || self.buffer.size < self.cfg.batchSize) return (float)-1.0;
    if (self.steps % self.cfg.trainEvery != 0UL) return (float)-1.0;
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    self.stepArena.begin();
    float l = dqn_train_(self);
    tensor_arena_end();
    self.stepArena.reset();
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    self.lastLoss = l;
    return l;
}

void Dqn::free() {
    self.opt.free();
    self.q.free();
    self.target.free();
    self.buffer.free();
    self.stepArena.free();
}

// ══ Policy gradient ══════════════════════════════════════════════════════════

struct PgConfig pg_config_default(int algo) {
    struct PgConfig c;
    c.algo = algo;
    c.hidden = 64UL;
    c.hiddenLayers = 2UL;
    c.act = ACT_TANH;
    c.gamma = (float)0.99;
    c.gaeLambda = (float)0.95;
    c.lr = algo == PG_REINFORCE ? (float)0.003 : (float)0.001;
    c.entropyCoef = (float)0.01;
    c.valueCoef = (float)0.5;
    c.clipEps = (float)0.2;
    c.epochs = 4UL;
    c.minibatch = 64UL;
    c.clipNorm = (float)0.5;
    c.normalizeAdv = 1;
    c.seed = 1UL;
    return c;
}

struct PolicyAgent pg_new(unsigned long obsDim, unsigned long actions, const &PgConfig cfg) {
    struct PolicyAgent a;
    a.cfg = *cfg;
    a.rng = rng_new(cfg.seed);
    a.policy = mlp_new(obsDim, cfg.hidden, cfg.hiddenLayers, actions, cfg.act, ACT_NONE, &a.rng);
    a.value = mlp_new(obsDim, cfg.hidden, cfg.hiddenLayers, 1UL, cfg.act, ACT_NONE, &a.rng);
    // small final policy layer: near-uniform initial policy
    &Tensor wl = *a.policy.params.at(2UL * (a.policy.nLayers - 1UL));
    unsafe {
        unsigned long i = 0UL;
        while (i < wl->size) { wl->data[i] = wl->data[i] * (float)0.01; i = i + 1UL; }
    }
    a.opt = optimizer_new(OPT_ADAM, cfg.lr);
    model_add_params(&a.policy, &a.opt);
    model_add_params(&a.value, &a.opt);
    a.stepArena = tensor_arena_new(0UL);
    a.obsDim = obsDim;
    a.actions = actions;
    a.rObs = vec_new(); a.rNext = vec_new(); a.rAction = vec_new();
    a.rReward = vec_new(); a.rTerminal = vec_new(); a.rDone = vec_new();
    a.rollout = 0UL;
    a.lastEntropy = (float)0.0;
    return a;
}

unsigned long PolicyAgent::act(const &Tensor obs, int sample) {
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    self.stepArena.begin();
    &Tensor logits = self.policy.forward(rl_row_(obs, self.obsDim));
    unsigned long choice = 0UL;
    unsafe {
        const float* l = (const float*)logits->data;
        unsigned long A = self.actions;
        if (!sample) {
            choice = rl_argmax_(l, A, &self.rng);
        } else {
            float mx = l[0];
            unsigned long i = 1UL;
            while (i < A) { if (l[i] > mx) mx = l[i]; i = i + 1UL; }
            float s = (float)0.0;
            i = 0UL;
            while (i < A) { s = s + exp_f(l[i] - mx); i = i + 1UL; }
            float u = self.rng.uniform() * s;
            float acc = (float)0.0;
            choice = A - 1UL;
            i = 0UL;
            while (i < A) { acc = acc + exp_f(l[i] - mx); if (u < acc) { choice = i; break; } i = i + 1UL; }
        }
    }
    tensor_arena_end();
    self.stepArena.reset();
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    return choice;
}

void PolicyAgent::record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                         int truncated) {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.obsDim) { self.rObs.push(obs->data[i]); self.rNext.push(next->data[i]); i = i + 1UL; }
    }
    self.rAction.push((float)action);
    self.rReward.push(reward);
    self.rTerminal.push(terminated ? (float)1.0 : (float)0.0);
    self.rDone.push(terminated || truncated ? (float)1.0 : (float)0.0);
    self.rollout = self.rollout + 1UL;
}

// Rows 'idx[0..m)' of a flat [N, w] float array as an [m, w] tensor.
static &Tensor pg_rows_(const float* src, unsigned long w, const unsigned long* idx, unsigned long m) {
    &Tensor t = tensor_new_2d(m, w, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < m) {
            unsigned long c = 0UL;
            while (c < w) { t->data[r * w + c] = src[idx[r] * w + c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    return t;
}

static float* pg_vec_ptr_(&Vec<float> v) {
    unsafe { return (float*)v.at(0UL); }
}

// One gradient step on the rows idx[0..m).
static float pg_step_(&PolicyAgent a, const float* obs, const float* act, const float* adv,
                      const float* ret, const float* logpOld, const unsigned long* idx, unsigned long m) {
    unsigned long A = a.actions;
    &Tensor O = pg_rows_(obs, a.obsDim, idx, m);
    &Tensor Ac = pg_rows_(act, 1UL, idx, m);
    &Tensor Adv = pg_rows_(adv, 1UL, idx, m);
    &Tensor Ret = pg_rows_(ret, 1UL, idx, m);
    &Tensor logpAll = tensor_log_softmax(a.policy.forward(O));
    &Tensor logp = tensor_sum_rows(tensor_mul(logpAll, tensor_one_hot(Ac, A)));
    &Tensor ent = tensor_scale(tensor_mean(tensor_sum_rows(tensor_mul(tensor_exp(logpAll), logpAll))), (float)-1.0);
    &Tensor pol = logp;
    if (a.cfg.algo == PG_PPO) {
        &Tensor Old = pg_rows_(logpOld, 1UL, idx, m);
        &Tensor ratio = tensor_exp(tensor_sub(logp, Old));
        &Tensor s1 = tensor_mul(ratio, Adv);
        &Tensor s2 = tensor_mul(tensor_clamp(ratio, (float)1.0 - a.cfg.clipEps, (float)1.0 + a.cfg.clipEps), Adv);
        pol = tensor_scale(tensor_mean(tensor_minimum(s1, s2)), (float)-1.0);
    } else {
        pol = tensor_scale(tensor_mean(tensor_mul(logp, Adv)), (float)-1.0);
    }
    &Tensor vl = tensor_mse_loss(a.value.forward(O), Ret);
    &Tensor loss = tensor_sub(tensor_add(pol, tensor_scale(vl, a.cfg.valueCoef)),
                              tensor_scale(ent, a.cfg.entropyCoef));
    a.opt.zero_grad();
    tensor_backward(loss);
    if (a.cfg.clipNorm > (float)0.0) a.opt.clip_grad_norm(a.cfg.clipNorm);
    a.opt.step();
    unsafe {
        a.lastEntropy = ent->data[0];
        return loss->data[0];
    }
}

float PolicyAgent::update() {
    unsigned long N = self.rollout;
    if (N == 0UL) return (float)0.0;
    unsigned long D = self.obsDim;
    float last = (float)0.0;
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    self.stepArena.begin();
    unsafe {
        const float* obs = (const float*)pg_vec_ptr_(&self.rObs);
        const float* nxt = (const float*)pg_vec_ptr_(&self.rNext);
        const float* act = (const float*)pg_vec_ptr_(&self.rAction);
        const float* rew = (const float*)pg_vec_ptr_(&self.rReward);
        const float* term = (const float*)pg_vec_ptr_(&self.rTerminal);
        const float* done = (const float*)pg_vec_ptr_(&self.rDone);
        float* adv = (float*)malloc(sizeof(float) * N);
        float* ret = (float*)malloc(sizeof(float) * N);
        float* logpOld = (float*)malloc(sizeof(float) * N);
        float* v = (float*)malloc(sizeof(float) * N);
        float* vn = (float*)malloc(sizeof(float) * N);
        unsigned long* idx = (unsigned long*)malloc(sizeof(unsigned long) * N);
        unsigned long i = 0UL;
        while (i < N) { idx[i] = i; i = i + 1UL; }

        // values, next values and old log-probabilities, without gradient
        tensor_set_grad_enabled(0);
        &Tensor O = pg_rows_(obs, D, (const unsigned long*)idx, N);
        &Tensor Nx = pg_rows_(nxt, D, (const unsigned long*)idx, N);
        &Tensor Ac = pg_rows_(act, 1UL, (const unsigned long*)idx, N);
        &Tensor Vt = self.value.forward(O);
        &Tensor Vn = self.value.forward(Nx);
        &Tensor Lp = tensor_sum_rows(tensor_mul(tensor_log_softmax(self.policy.forward(O)), tensor_one_hot(Ac, self.actions)));
        i = 0UL;
        while (i < N) { v[i] = Vt->data[i]; vn[i] = Vn->data[i]; logpOld[i] = Lp->data[i]; i = i + 1UL; }
        self.stepArena.reset();
        tensor_set_grad_enabled(1);

        float g = self.cfg.gamma;
        if (self.cfg.algo == PG_REINFORCE) {
            // Monte Carlo returns; a cut-off episode bootstraps from V(next)
            float G = (float)0.0;
            long t = (long)N - 1L;
            while (t >= 0L) {
                unsigned long u = (unsigned long)t;
                float boot = term[u] > (float)0.5 ? (float)0.0 : vn[u];
                if (done[u] > (float)0.5 || u == N - 1UL) G = rew[u] + g * boot;
                else G = rew[u] + g * G;
                ret[u] = G;
                adv[u] = G - v[u];
                t = t - 1L;
            }
        } else {
            // GAE(lambda)
            float lam = self.cfg.gaeLambda;
            float gae = (float)0.0;
            long t = (long)N - 1L;
            while (t >= 0L) {
                unsigned long u = (unsigned long)t;
                float notTerm = term[u] > (float)0.5 ? (float)0.0 : (float)1.0;
                float notDone = done[u] > (float)0.5 ? (float)0.0 : (float)1.0;
                float delta = rew[u] + g * notTerm * vn[u] - v[u];
                gae = delta + g * lam * notDone * gae;
                adv[u] = gae;
                ret[u] = gae + v[u];
                t = t - 1L;
            }
        }
        if (self.cfg.normalizeAdv && N > 1UL) {
            double s = 0.0; double sq = 0.0;
            i = 0UL;
            while (i < N) { s = s + (double)adv[i]; sq = sq + (double)adv[i] * (double)adv[i]; i = i + 1UL; }
            double m = s / (double)N;
            double sd = sqrt_d(sq / (double)N - m * m) + 1e-8;
            i = 0UL;
            while (i < N) { adv[i] = (float)(((double)adv[i] - m) / sd); i = i + 1UL; }
        }

        if (self.cfg.algo == PG_PPO) {
            unsigned long mb = self.cfg.minibatch == 0UL ? N : self.cfg.minibatch;
            unsigned long e = 0UL;
            while (e < self.cfg.epochs) {
                // shuffle, then minibatches
                i = N;
                while (i > 1UL) {
                    unsigned long j = self.rng.below(i);
                    unsigned long tmp = idx[i - 1UL]; idx[i - 1UL] = idx[j]; idx[j] = tmp;
                    i = i - 1UL;
                }
                unsigned long start = 0UL;
                while (start < N) {
                    unsigned long m = N - start < mb ? N - start : mb;
                    last = pg_step_(self, obs, act, (const float*)adv, (const float*)ret, (const float*)logpOld,
                                    (const unsigned long*)&idx[start], m);
                    self.stepArena.reset();
                    start = start + m;
                }
                e = e + 1UL;
            }
        } else {
            last = pg_step_(self, obs, act, (const float*)adv, (const float*)ret, (const float*)logpOld,
                            (const unsigned long*)idx, N);
            self.stepArena.reset();
        }
        free((void*)adv); free((void*)ret); free((void*)logpOld);
        free((void*)v); free((void*)vn); free((void*)idx);
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    self.rObs.clear(); self.rNext.clear(); self.rAction.clear();
    self.rReward.clear(); self.rTerminal.clear(); self.rDone.clear();
    self.rollout = 0UL;
    return last;
}

void PolicyAgent::free() {
    self.opt.free();
    self.policy.free();
    self.value.free();
    self.stepArena.free();
    self.rObs.free(); self.rNext.free(); self.rAction.free();
    self.rReward.free(); self.rTerminal.free(); self.rDone.free();
}

void Dqn::collect_params(&Vec<&Tensor> out) const { self.q.collect_params(out); }

void Dqn::collect_state(&Vec<&Tensor> out) const {
    self.q.collect_state(out);
    self.target.collect_state(out);
}

void PolicyAgent::collect_params(&Vec<&Tensor> out) const {
    self.policy.collect_params(out);
    self.value.collect_params(out);
}

void PolicyAgent::collect_state(&Vec<&Tensor> out) const { self.collect_params(out); }

} // namespace std
