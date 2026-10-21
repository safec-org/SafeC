#pragma once
// SafeC Standard Library — reinforcement learning.
//
// Environments:
//   Bandit      k-armed Gaussian bandit.
//   GridWorld   tabular grid with walls, pits and goals; gridworld_cliff()
//               is Sutton & Barto's cliff walk (Example 6.6).
//   CartPole    the classic pole-balancing control task (Barto et al.
//               1983, Gym's CartPole-v1 dynamics and limits).
//
// Agents, grouped by technique:
//   Bandits        epsilon-greedy, UCB1, Thompson sampling, gradient bandit.
//   Tabular TD     Q-learning, SARSA, expected SARSA.
//   Value-based    DQN (Mnih et al. 2015): replay buffer, target network,
//                  Huber loss, epsilon schedule, optional Double DQN.
//   Policy-based   REINFORCE with a learned baseline, A2C (advantage
//                  actor-critic with GAE) and PPO (clipped surrogate,
//                  several epochs of minibatches per rollout).
//
// Observations are tensors (an agent reads obs->data[0 .. obsDim)), and
// states and actions of discrete problems are indices. A typical loop:
//   struct CartPole env = cartpole_new(1UL);
//   &Tensor obs = tensor_new_1d(4UL, 0);  &Tensor next = tensor_new_1d(4UL, 0);
//   cartpole_reset(&env, obs);
//   unsigned long a = dqn_act(&agent, obs, 1);
//   struct EnvStep s = cartpole_step(&env, a, next);
//   dqn_observe(&agent, obs, a, s.reward, next, s.terminated);
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/collections/vec.h>

namespace std {

// What a step returned: 'terminated' ends the episode in a terminal state
// (no future reward); 'truncated' cuts it off by a time limit (the state
// still has value, and learners bootstrap from it).
struct EnvStep {
    float reward;
    int   terminated;
    int   truncated;
};

// ── Bandit ───────────────────────────────────────────────────────────────────
struct Bandit {
    &Tensor means;           // [k], drawn from N(0, 1)
    unsigned long k;
    struct Rng rng;
};
struct Bandit  bandit_new(unsigned long k, unsigned long seed);
float          bandit_pull(&Bandit b, unsigned long arm);     // N(means[arm], 1)
unsigned long  bandit_best_arm(const &Bandit b);
void           bandit_free(&Bandit b);

// ── GridWorld ────────────────────────────────────────────────────────────────
#define GRID_EMPTY 0
#define GRID_WALL  1   // cannot be entered
#define GRID_PIT   2   // pitReward; back to the start (or the episode ends when pitTerminal)
#define GRID_GOAL  3   // goalReward, episode ends

#define GRID_UP    0
#define GRID_RIGHT 1
#define GRID_DOWN  2
#define GRID_LEFT  3

struct GridWorld {
    &Tensor cells;           // [height * width] GRID_* tags, row 0 at the top
    unsigned long width;
    unsigned long height;
    unsigned long start;     // state index y * width + x
    unsigned long state;
    float stepReward;        // every move (default -1)
    float pitReward;         // default -100
    float goalReward;        // default 0
    int   pitTerminal;       // default 0 (cliff walk: back to start)
    float slip;              // probability the move goes a random way (0)
    unsigned long steps;
    unsigned long maxSteps;  // truncation (default 1000)
    struct Rng rng;
};
// An empty w x h grid, start top-left, goal bottom-right.
struct GridWorld gridworld_new(unsigned long width, unsigned long height, unsigned long seed);
// The 12 x 4 cliff walk: start bottom-left, goal bottom-right, the bottom
// row between them a cliff (-100, back to start).
struct GridWorld gridworld_cliff(unsigned long seed);
void gridworld_set(&GridWorld g, unsigned long x, unsigned long y, int cell);
void gridworld_set_start(&GridWorld g, unsigned long x, unsigned long y);
unsigned long gridworld_states(const &GridWorld g);
unsigned long gridworld_reset(&GridWorld g);                        // -> start state
struct EnvStep gridworld_step(&GridWorld g, unsigned long action, &unsigned long next);
// One-hot observation of a state (for neural agents), obs has states() values.
void gridworld_one_hot(const &GridWorld g, unsigned long state, &Tensor obs);
void gridworld_free(&GridWorld g);

// ── CartPole ─────────────────────────────────────────────────────────────────
#define CARTPOLE_OBS     4UL   // cart position, cart velocity, pole angle, pole angular velocity
#define CARTPOLE_ACTIONS 2UL   // push left, push right
struct CartPole {
    float x;
    float xDot;
    float theta;
    float thetaDot;
    unsigned long steps;
    unsigned long maxSteps;  // 500 (CartPole-v1)
    struct Rng rng;
};
struct CartPole cartpole_new(unsigned long seed);
void            cartpole_reset(&CartPole env, &Tensor obs);
// Reward 1 per step; terminated when |x| > 2.4 or |theta| > 12 degrees,
// truncated at maxSteps.
struct EnvStep  cartpole_step(&CartPole env, unsigned long action, &Tensor obs);

// ══ Bandit agents ════════════════════════════════════════════════════════════
#define BANDIT_EPS_GREEDY 0   // param: epsilon
#define BANDIT_UCB1       1   // param: exploration c (sqrt(2) in Auer et al.)
#define BANDIT_THOMPSON   2   // Gaussian posterior per arm; param: prior std
#define BANDIT_GRADIENT   3   // softmax preferences with a reward baseline; param: step size

struct BanditAgent {
    int kind;
    unsigned long k;
    &Tensor counts;          // pulls per arm
    &Tensor values;          // sample means (preferences for BANDIT_GRADIENT)
    float param;
    float baseline;          // mean reward so far
    unsigned long t;
    struct Rng rng;
};
struct BanditAgent bandit_agent_new(int kind, unsigned long k, float param, unsigned long seed);
unsigned long      bandit_agent_select(&BanditAgent a);
void               bandit_agent_update(&BanditAgent a, unsigned long arm, float reward);
void               bandit_agent_free(&BanditAgent a);

// ══ Tabular TD control ═══════════════════════════════════════════════════════
struct TabularAgent {
    &Tensor q;               // [states, actions]
    unsigned long states;
    unsigned long actions;
    float alpha;             // step size
    float gamma;             // discount
    float epsilon;           // exploration (change it to decay)
    struct Rng rng;
};
struct TabularAgent tabular_agent_new(unsigned long states, unsigned long actions,
                                      float alpha, float gamma, float epsilon, unsigned long seed);
unsigned long tabular_act(&TabularAgent a, unsigned long state);       // epsilon-greedy
unsigned long tabular_greedy(const &TabularAgent a, unsigned long state);
// Each returns the TD error. 'terminal' drops the bootstrap term.
// Q-learning (off-policy): target r + gamma max_a' Q(s', a').
float q_learning_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                        unsigned long s2, int terminal);
// SARSA (on-policy): target r + gamma Q(s', a') for the a' actually taken next.
float sarsa_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                   unsigned long s2, unsigned long a2, int terminal);
// Expected SARSA: target r + gamma E_pi[Q(s', .)] under the epsilon-greedy policy.
float expected_sarsa_update(&TabularAgent a, unsigned long s, unsigned long act, float r,
                            unsigned long s2, int terminal);
void tabular_agent_free(&TabularAgent a);

// ══ DQN ══════════════════════════════════════════════════════════════════════
struct ReplayBuffer {
    &Tensor obs;             // [slots, obsDim]
    &Tensor next;            // [slots, obsDim]
    &Tensor action;          // [slots]
    &Tensor reward;          // [slots]
    &Tensor terminal;        // [slots]
    unsigned long slots;
    unsigned long size;
    unsigned long pos;
    unsigned long obsDim;
};
struct ReplayBuffer replay_new(unsigned long slots, unsigned long obsDim);
void replay_push(&ReplayBuffer rb, const &Tensor obs, unsigned long action, float reward,
                 const &Tensor next, int terminal);
void replay_free(&ReplayBuffer rb);

struct DqnConfig {
    unsigned long hidden;
    unsigned long hiddenLayers;
    float gamma;
    float lr;
    unsigned long batchSize;
    unsigned long bufferSize;
    unsigned long learningStarts;   // transitions before the first update
    unsigned long trainEvery;       // environment steps per gradient step
    unsigned long targetEvery;      // hard target copy every N gradient steps; 0: soft updates
    float tau;                      // soft target update rate (targetEvery 0)
    float epsStart;
    float epsEnd;
    unsigned long epsDecaySteps;    // linear decay over this many steps
    int   doubleDqn;                // van Hasselt et al. 2016: online net picks a', target net scores it
    float huberDelta;
    float clipNorm;                 // gradient norm clip, 0 = off
    unsigned long seed;
};
// 2 hidden layers of 64 (ReLU), gamma 0.99, Adam lr 5e-4, batch 64, buffer
// 50000, learning after 1000 steps, train every step, soft target updates
// (tau 0.005), epsilon 1 -> 0.05 over 10000 steps, Double DQN, Huber 1,
// clip 10.
struct DqnConfig dqn_config_default();

struct Dqn {
    struct Mlp q;
    struct Mlp target;
    struct Optimizer opt;
    struct ReplayBuffer buffer;
    struct DqnConfig cfg;
    struct Rng rng;
    struct TensorArena stepArena;
    unsigned long obsDim;
    unsigned long actions;
    unsigned long steps;            // environment steps observed
    unsigned long updates;          // gradient steps taken
    float lastLoss;
};
struct Dqn    dqn_new(unsigned long obsDim, unsigned long actions, const &DqnConfig cfg);
float         dqn_epsilon(const &Dqn d);
// explore 1: epsilon-greedy with the current epsilon; 0: greedy.
unsigned long dqn_act(&Dqn d, const &Tensor obs, int explore);
// Stores the transition and trains when due; returns the loss, or -1 when
// no gradient step ran.
float         dqn_observe(&Dqn d, const &Tensor obs, unsigned long action, float reward,
                          const &Tensor next, int terminal);
void          dqn_free(&Dqn d);

// ══ Policy gradient: REINFORCE, A2C, PPO ═════════════════════════════════════
#define PG_REINFORCE 0
#define PG_A2C       1
#define PG_PPO       2

struct PgConfig {
    int   algo;                     // PG_*
    unsigned long hidden;
    unsigned long hiddenLayers;
    int   act;
    float gamma;
    float gaeLambda;                // A2C / PPO advantages (1 = Monte Carlo, 0 = one-step TD)
    float lr;
    float entropyCoef;
    float valueCoef;
    float clipEps;                  // PPO ratio clip
    unsigned long epochs;           // PPO passes over a rollout
    unsigned long minibatch;        // PPO minibatch size
    float clipNorm;                 // 0 = off
    int   normalizeAdv;
    unsigned long seed;
};
// 2 hidden layers of 64 (tanh), gamma 0.99, lambda 0.95, Adam lr 1e-3
// (3e-3 for REINFORCE), entropy 0.01, value 0.5, clip 0.2, 4 epochs of 64,
// grad clip 0.5, normalized advantages.
struct PgConfig pg_config_default(int algo);

// Separate policy (logits) and value networks, one Adam optimizer. The
// rollout collects transitions until pg_update consumes them: call it at
// episode ends (REINFORCE) or every N steps (A2C / PPO).
struct PolicyAgent {
    struct Mlp policy;
    struct Mlp value;
    struct Optimizer opt;
    struct PgConfig cfg;
    struct Rng rng;
    struct TensorArena stepArena;
    unsigned long obsDim;
    unsigned long actions;
    struct Vec<float> rObs;
    struct Vec<float> rNext;
    struct Vec<float> rAction;
    struct Vec<float> rReward;
    struct Vec<float> rTerminal;
    struct Vec<float> rDone;        // terminated or truncated
    unsigned long rollout;          // transitions stored
    float lastEntropy;
};
struct PolicyAgent pg_new(unsigned long obsDim, unsigned long actions, const &PgConfig cfg);
// sample 1: draw from the policy; 0: the most likely action.
unsigned long pg_act(&PolicyAgent a, const &Tensor obs, int sample);
void  pg_record(&PolicyAgent a, const &Tensor obs, unsigned long action, float reward,
                const &Tensor next, int terminated, int truncated);
// Trains on the stored rollout with the configured algorithm, clears it,
// and returns the last loss.
float pg_update(&PolicyAgent a);
void  pg_free(&PolicyAgent a);

} // namespace std
