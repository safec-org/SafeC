// SafeC Standard Library — reinforcement learning (see rl.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/rl.h>
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
        case some(a): { tensor_arena_begin(a); }
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
                if (rng_below(rng, ties) == 0UL) best = i;
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
    rng_fill_normal(&b.rng, b.means, (float)0.0, (float)1.0);
    return b;
}

float bandit_pull(&Bandit b, unsigned long arm) {
    unsafe { return b.means->data[arm] + rng_normal(&b.rng); }
}

unsigned long bandit_best_arm(const &Bandit b) {
    unsigned long best = 0UL;
    unsafe {
        unsigned long i = 1UL;
        while (i < b.k) { if (b.means->data[i] > b.means->data[best]) best = i; i = i + 1UL; }
    }
    return best;
}

void bandit_free(&Bandit b) { rl_drop_(b.means); }

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
    gridworld_set(&g, width - 1UL, height - 1UL, GRID_GOAL);
    return g;
}

struct GridWorld gridworld_cliff(unsigned long seed) {
    struct GridWorld g = gridworld_new(12UL, 4UL, seed);
    gridworld_set_start(&g, 0UL, 3UL);
    unsigned long x = 1UL;
    while (x < 11UL) { gridworld_set(&g, x, 3UL, GRID_PIT); x = x + 1UL; }
    return g;
}

void gridworld_set(&GridWorld g, unsigned long x, unsigned long y, int cell) {
    unsafe { g.cells->data[y * g.width + x] = (float)cell; }
}

void gridworld_set_start(&GridWorld g, unsigned long x, unsigned long y) {
    g.start = y * g.width + x;
}

unsigned long gridworld_states(const &GridWorld g) { return g.width * g.height; }

unsigned long gridworld_reset(&GridWorld g) {
    g.state = g.start;
    g.steps = 0UL;
    return g.state;
}

static int grid_cell_(const &GridWorld g, unsigned long s) {
    unsafe { return (int)g.cells->data[s]; }
}

struct EnvStep gridworld_step(&GridWorld g, unsigned long action, &unsigned long next) {
    struct EnvStep r;
    r.reward = g.stepReward;
    r.terminated = 0;
    r.truncated = 0;
    if (g.slip > (float)0.0 && rng_uniform(&g.rng) < g.slip) action = rng_below(&g.rng, 4UL);
    unsigned long x = g.state % g.width;
    unsigned long y = g.state / g.width;
    if (action == GRID_UP && y > 0UL) y = y - 1UL;
    else if (action == GRID_RIGHT && x + 1UL < g.width) x = x + 1UL;
    else if (action == GRID_DOWN && y + 1UL < g.height) y = y + 1UL;
    else if (action == GRID_LEFT && x > 0UL) x = x - 1UL;
    unsigned long s = y * g.width + x;
    int cell = grid_cell_(g, s);
    if (cell != GRID_WALL) g.state = s;
    if (cell == GRID_PIT) {
        r.reward = g.pitReward;
        if (g.pitTerminal) r.terminated = 1;
        else g.state = g.start;
    } else if (cell == GRID_GOAL) {
        r.reward = r.reward + g.goalReward;
        r.terminated = 1;
    }
    g.steps = g.steps + 1UL;
    if (!r.terminated && g.steps >= g.maxSteps) r.truncated = 1;
    *next = g.state;
    return r;
}

void gridworld_one_hot(const &GridWorld g, unsigned long state, &Tensor obs) {
    unsafe {
        unsigned long n = g.width * g.height;
        unsigned long i = 0UL;
        while (i < n) { obs->data[i] = i == state ? (float)1.0 : (float)0.0; i = i + 1UL; }
    }
}

void gridworld_free(&GridWorld g) { rl_drop_(g.cells); }

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

void cartpole_reset(&CartPole env, &Tensor obs) {
    env.x = (float)0.1 * rng_uniform(&env.rng) - (float)0.05;
    env.xDot = (float)0.1 * rng_uniform(&env.rng) - (float)0.05;
    env.theta = (float)0.1 * rng_uniform(&env.rng) - (float)0.05;
    env.thetaDot = (float)0.1 * rng_uniform(&env.rng) - (float)0.05;
    env.steps = 0UL;
    cartpole_obs_(env, obs);
}

struct EnvStep cartpole_step(&CartPole env, unsigned long action, &Tensor obs) {
    float gravity = (float)9.8;
    float massPole = (float)0.1;
    float totalMass = (float)1.1;
    float length = (float)0.5;            // half the pole
    float poleMassLength = (float)0.05;
    float tau = (float)0.02;
    float force = action == 1UL ? (float)10.0 : (float)-10.0;
    float c = cos_f(env.theta);
    float s = sin_f(env.theta);
    float temp = (force + poleMassLength * env.thetaDot * env.thetaDot * s) / totalMass;
    float thetaAcc = (gravity * s - c * temp) /
                     (length * ((float)4.0 / (float)3.0 - massPole * c * c / totalMass));
    float xAcc = temp - poleMassLength * thetaAcc * c / totalMass;
    env.x = env.x + tau * env.xDot;
    env.xDot = env.xDot + tau * xAcc;
    env.theta = env.theta + tau * env.thetaDot;
    env.thetaDot = env.thetaDot + tau * thetaAcc;
    env.steps = env.steps + 1UL;
    cartpole_obs_(env, obs);
    struct EnvStep r;
    r.reward = (float)1.0;
    float lim = (float)0.20943951;        // 12 degrees
    r.terminated = env.x < (float)-2.4 || env.x > (float)2.4 || env.theta < -lim || env.theta > lim;
    r.truncated = !r.terminated && env.steps >= env.maxSteps;
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

unsigned long bandit_agent_select(&BanditAgent a) {
    unsigned long k = a.k;
    unsigned long choice = 0UL;
    unsafe {
        float* score = (float*)malloc(sizeof(float) * k);
        const float* n = (const float*)a.counts->data;
        const float* q = (const float*)a.values->data;
        unsigned long i = 0UL;
        if (a.kind == BANDIT_EPS_GREEDY) {
            if (rng_uniform(&a.rng) < a.param) choice = rng_below(&a.rng, k);
            else choice = rl_argmax_(q, k, &a.rng);
        } else if (a.kind == BANDIT_UCB1) {
            int untried = 0;
            while (i < k) { if (n[i] == (float)0.0) { untried = 1; choice = i; break; } i = i + 1UL; }
            if (!untried) {
                float lt = log_f((float)a.t);
                i = 0UL;
                while (i < k) { score[i] = q[i] + a.param * sqrt_f(lt / n[i]); i = i + 1UL; }
                choice = rl_argmax_((const float*)score, k, &a.rng);
            }
        } else if (a.kind == BANDIT_THOMPSON) {
            // posterior of each arm's mean under a N(0, param^2) prior and unit noise
            float prior = (float)1.0 / (a.param * a.param);
            while (i < k) {
                float prec = prior + n[i];
                score[i] = n[i] * q[i] / prec + rng_normal(&a.rng) / sqrt_f(prec);
                i = i + 1UL;
            }
            choice = rl_argmax_((const float*)score, k, &a.rng);
        } else {
            bandit_softmax_(a, score);
            float u = rng_uniform(&a.rng);
            float acc = (float)0.0;
            choice = k - 1UL;
            while (i < k) { acc = acc + score[i]; if (u < acc) { choice = i; break; } i = i + 1UL; }
        }
        free((void*)score);
    }
    return choice;
}

void bandit_agent_update(&BanditAgent a, unsigned long arm, float reward) {
    a.t = a.t + 1UL;
    unsafe {
        a.counts->data[arm] = a.counts->data[arm] + (float)1.0;
        if (a.kind == BANDIT_GRADIENT) {
            float* p = (float*)malloc(sizeof(float) * a.k);
            bandit_softmax_(a, p);
            float d = reward - a.baseline;
            unsigned long i = 0UL;
            while (i < a.k) {
                float ind = i == arm ? (float)1.0 : (float)0.0;
                a.values->data[i] = a.values->data[i] + a.param * d * (ind - p[i]);
                i = i + 1UL;
            }
            free((void*)p);
        } else {
            float n = a.counts->data[arm];
            a.values->data[arm] = a.values->data[arm] + (reward - a.values->data[arm]) / n;
        }
    }
    a.baseline = a.baseline + (reward - a.baseline) / (float)a.t;
}

void bandit_agent_free(&BanditAgent a) {
    rl_drop_(a.counts);
    rl_drop_(a.values);
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

unsigned long tabular_greedy(const &TabularAgent a, unsigned long state) {
    unsigned long best = 0UL;
    unsigned long i = 1UL;
    while (i < a.actions) { if (tab_q_(a, state, i) > tab_q_(a, state, best)) best = i; i = i + 1UL; }
    return best;
}

unsigned long tabular_act(&TabularAgent a, unsigned long state) {
    if (rng_uniform(&a.rng) < a.epsilon) return rng_below(&a.rng, a.actions);
    unsafe { return rl_argmax_((const float*)&a.q->data[state * a.actions], a.actions, &a.rng); }
}

static float tab_apply_(&TabularAgent a, unsigned long s, unsigned long act, float target) {
    float td = target - tab_q_(a, s, act);
    unsafe { a.q->data[s * a.actions + act] = a.q->data[s * a.actions + act] + a.alpha * td; }
    return td;
}

float q_learning_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                        unsigned long s2, int terminal) {
    float target = terminal ? r : r + a.gamma * tab_max_(a, s2);
    return tab_apply_(a, s, act, target);
}

float sarsa_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                   unsigned long s2, unsigned long a2, int terminal) {
    float target = terminal ? r : r + a.gamma * tab_q_(a, s2, a2);
    return tab_apply_(a, s, act, target);
}

float expected_sarsa_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                            unsigned long s2, int terminal) {
    float target = r;
    if (!terminal) {
        float mean = (float)0.0;
        unsigned long i = 0UL;
        while (i < a.actions) { mean = mean + tab_q_(a, s2, i); i = i + 1UL; }
        mean = mean / (float)a.actions;
        float ev = ((float)1.0 - a.epsilon) * tab_max_(a, s2) + a.epsilon * mean;
        target = r + a.gamma * ev;
    }
    return tab_apply_(a, s, act, target);
}

void tabular_agent_free(&TabularAgent a) { rl_drop_(a.q); }

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

void replay_push(&ReplayBuffer rb, const &Tensor obs, unsigned long action, float reward,
                 const &Tensor next, int terminal) {
    unsigned long p = rb.pos;
    unsigned long d = rb.obsDim;
    unsafe {
        unsigned long i = 0UL;
        while (i < d) {
            rb.obs->data[p * d + i] = obs->data[i];
            rb.next->data[p * d + i] = next->data[i];
            i = i + 1UL;
        }
        rb.action->data[p] = (float)action;
        rb.reward->data[p] = reward;
        rb.terminal->data[p] = terminal ? (float)1.0 : (float)0.0;
    }
    rb.pos = (p + 1UL) % rb.slots;
    if (rb.size < rb.slots) rb.size = rb.size + 1UL;
}

void replay_free(&ReplayBuffer rb) {
    rl_drop_(rb.obs); rl_drop_(rb.next); rl_drop_(rb.action); rl_drop_(rb.reward); rl_drop_(rb.terminal);
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
    mlp_copy(&d.target, &d.q);
    d.opt = optimizer_new(OPT_ADAM, cfg.lr);
    optimizer_add_mlp(&d.opt, &d.q);
    d.buffer = replay_new(cfg.bufferSize, obsDim);
    d.stepArena = tensor_arena_new(0UL);
    d.obsDim = obsDim;
    d.actions = actions;
    d.steps = 0UL;
    d.updates = 0UL;
    d.lastLoss = (float)0.0;
    return d;
}

float dqn_epsilon(const &Dqn d) {
    if (d.steps >= d.cfg.epsDecaySteps) return d.cfg.epsEnd;
    float f = (float)d.steps / (float)d.cfg.epsDecaySteps;
    return d.cfg.epsStart + f * (d.cfg.epsEnd - d.cfg.epsStart);
}

unsigned long dqn_act(&Dqn d, const &Tensor obs, int explore) {
    if (explore && rng_uniform(&d.rng) < dqn_epsilon(d)) return rng_below(&d.rng, d.actions);
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    tensor_arena_begin(&d.stepArena);
    &Tensor q = mlp_forward(&d.q, rl_row_(obs, d.obsDim));
    unsigned long a;
    unsafe { a = rl_argmax_((const float*)q->data, d.actions, &d.rng); }
    tensor_arena_end();
    tensor_arena_reset(&d.stepArena);
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
            unsigned long j = rng_below(&d.rng, d.buffer.size);
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
    &Tensor qt = mlp_forward(&d.target, nx);
    &Tensor qo = d.cfg.doubleDqn ? mlp_forward(&d.q, nx) : qt;
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
    &Tensor qsa = tensor_sum_rows(tensor_mul(mlp_forward(&d.q, ob), tensor_one_hot(act, A)));
    &Tensor loss = tensor_huber_loss(qsa, y, d.cfg.huberDelta);
    optimizer_zero_grad(&d.opt);
    tensor_backward(loss);
    if (d.cfg.clipNorm > (float)0.0) optimizer_clip_grad_norm(&d.opt, d.cfg.clipNorm);
    optimizer_step(&d.opt);
    d.updates = d.updates + 1UL;
    if (d.cfg.targetEvery > 0UL) {
        if (d.updates % d.cfg.targetEvery == 0UL) mlp_copy(&d.target, &d.q);
    } else {
        mlp_soft_update(&d.target, &d.q, d.cfg.tau);
    }
    unsafe { return loss->data[0]; }
}

float dqn_observe(&Dqn d, const &Tensor obs, unsigned long action, float reward,
                  const &Tensor next, int terminal) {
    replay_push(&d.buffer, obs, action, reward, next, terminal);
    d.steps = d.steps + 1UL;
    if (d.buffer.size < d.cfg.learningStarts || d.buffer.size < d.cfg.batchSize) return (float)-1.0;
    if (d.steps % d.cfg.trainEvery != 0UL) return (float)-1.0;
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_arena_begin(&d.stepArena);
    float l = dqn_train_(d);
    tensor_arena_end();
    tensor_arena_reset(&d.stepArena);
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    d.lastLoss = l;
    return l;
}

void dqn_free(&Dqn d) {
    optimizer_free(&d.opt);
    mlp_free(&d.q);
    mlp_free(&d.target);
    replay_free(&d.buffer);
    tensor_arena_free(&d.stepArena);
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
    optimizer_add_mlp(&a.opt, &a.policy);
    optimizer_add_mlp(&a.opt, &a.value);
    a.stepArena = tensor_arena_new(0UL);
    a.obsDim = obsDim;
    a.actions = actions;
    a.rObs = vec_new(); a.rNext = vec_new(); a.rAction = vec_new();
    a.rReward = vec_new(); a.rTerminal = vec_new(); a.rDone = vec_new();
    a.rollout = 0UL;
    a.lastEntropy = (float)0.0;
    return a;
}

unsigned long pg_act(&PolicyAgent a, const &Tensor obs, int sample) {
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    tensor_arena_begin(&a.stepArena);
    &Tensor logits = mlp_forward(&a.policy, rl_row_(obs, a.obsDim));
    unsigned long choice = 0UL;
    unsafe {
        const float* l = (const float*)logits->data;
        unsigned long A = a.actions;
        if (!sample) {
            choice = rl_argmax_(l, A, &a.rng);
        } else {
            float mx = l[0];
            unsigned long i = 1UL;
            while (i < A) { if (l[i] > mx) mx = l[i]; i = i + 1UL; }
            float s = (float)0.0;
            i = 0UL;
            while (i < A) { s = s + exp_f(l[i] - mx); i = i + 1UL; }
            float u = rng_uniform(&a.rng) * s;
            float acc = (float)0.0;
            choice = A - 1UL;
            i = 0UL;
            while (i < A) { acc = acc + exp_f(l[i] - mx); if (u < acc) { choice = i; break; } i = i + 1UL; }
        }
    }
    tensor_arena_end();
    tensor_arena_reset(&a.stepArena);
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    return choice;
}

void pg_record(&PolicyAgent a, const &Tensor obs, unsigned long action, float reward,
               const &Tensor next, int terminated, int truncated) {
    unsafe {
        unsigned long i = 0UL;
        while (i < a.obsDim) { a.rObs.push(obs->data[i]); a.rNext.push(next->data[i]); i = i + 1UL; }
    }
    a.rAction.push((float)action);
    a.rReward.push(reward);
    a.rTerminal.push(terminated ? (float)1.0 : (float)0.0);
    a.rDone.push(terminated || truncated ? (float)1.0 : (float)0.0);
    a.rollout = a.rollout + 1UL;
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
    &Tensor logpAll = tensor_log_softmax(mlp_forward(&a.policy, O));
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
    &Tensor vl = tensor_mse_loss(mlp_forward(&a.value, O), Ret);
    &Tensor loss = tensor_sub(tensor_add(pol, tensor_scale(vl, a.cfg.valueCoef)),
                              tensor_scale(ent, a.cfg.entropyCoef));
    optimizer_zero_grad(&a.opt);
    tensor_backward(loss);
    if (a.cfg.clipNorm > (float)0.0) optimizer_clip_grad_norm(&a.opt, a.cfg.clipNorm);
    optimizer_step(&a.opt);
    unsafe {
        a.lastEntropy = ent->data[0];
        return loss->data[0];
    }
}

float pg_update(&PolicyAgent a) {
    unsigned long N = a.rollout;
    if (N == 0UL) return (float)0.0;
    unsigned long D = a.obsDim;
    float last = (float)0.0;
    ?&TensorArena outer = tensor_arena_current_();
    int was = tensor_is_grad_enabled();
    tensor_arena_begin(&a.stepArena);
    unsafe {
        const float* obs = (const float*)pg_vec_ptr_(&a.rObs);
        const float* nxt = (const float*)pg_vec_ptr_(&a.rNext);
        const float* act = (const float*)pg_vec_ptr_(&a.rAction);
        const float* rew = (const float*)pg_vec_ptr_(&a.rReward);
        const float* term = (const float*)pg_vec_ptr_(&a.rTerminal);
        const float* done = (const float*)pg_vec_ptr_(&a.rDone);
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
        &Tensor Vt = mlp_forward(&a.value, O);
        &Tensor Vn = mlp_forward(&a.value, Nx);
        &Tensor Lp = tensor_sum_rows(tensor_mul(tensor_log_softmax(mlp_forward(&a.policy, O)), tensor_one_hot(Ac, a.actions)));
        i = 0UL;
        while (i < N) { v[i] = Vt->data[i]; vn[i] = Vn->data[i]; logpOld[i] = Lp->data[i]; i = i + 1UL; }
        tensor_arena_reset(&a.stepArena);
        tensor_set_grad_enabled(1);

        float g = a.cfg.gamma;
        if (a.cfg.algo == PG_REINFORCE) {
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
            float lam = a.cfg.gaeLambda;
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
        if (a.cfg.normalizeAdv && N > 1UL) {
            double s = 0.0; double sq = 0.0;
            i = 0UL;
            while (i < N) { s = s + (double)adv[i]; sq = sq + (double)adv[i] * (double)adv[i]; i = i + 1UL; }
            double m = s / (double)N;
            double sd = sqrt_d(sq / (double)N - m * m) + 1e-8;
            i = 0UL;
            while (i < N) { adv[i] = (float)(((double)adv[i] - m) / sd); i = i + 1UL; }
        }

        if (a.cfg.algo == PG_PPO) {
            unsigned long mb = a.cfg.minibatch == 0UL ? N : a.cfg.minibatch;
            unsigned long e = 0UL;
            while (e < a.cfg.epochs) {
                // shuffle, then minibatches
                i = N;
                while (i > 1UL) {
                    unsigned long j = rng_below(&a.rng, i);
                    unsigned long tmp = idx[i - 1UL]; idx[i - 1UL] = idx[j]; idx[j] = tmp;
                    i = i - 1UL;
                }
                unsigned long start = 0UL;
                while (start < N) {
                    unsigned long m = N - start < mb ? N - start : mb;
                    last = pg_step_(a, obs, act, (const float*)adv, (const float*)ret, (const float*)logpOld,
                                    (const unsigned long*)&idx[start], m);
                    tensor_arena_reset(&a.stepArena);
                    start = start + m;
                }
                e = e + 1UL;
            }
        } else {
            last = pg_step_(a, obs, act, (const float*)adv, (const float*)ret, (const float*)logpOld,
                            (const unsigned long*)idx, N);
            tensor_arena_reset(&a.stepArena);
        }
        free((void*)adv); free((void*)ret); free((void*)logpOld);
        free((void*)v); free((void*)vn); free((void*)idx);
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    rl_heap_end_(outer);
    a.rObs.clear(); a.rNext.clear(); a.rAction.clear();
    a.rReward.clear(); a.rTerminal.clear(); a.rDone.clear();
    a.rollout = 0UL;
    return last;
}

void pg_free(&PolicyAgent a) {
    optimizer_free(&a.opt);
    mlp_free(&a.policy);
    mlp_free(&a.value);
    tensor_arena_free(&a.stepArena);
    a.rObs.free(); a.rNext.free(); a.rAction.free();
    a.rReward.free(); a.rTerminal.free(); a.rDone.free();
}

} // namespace std
