## 1. Introduction

This project explores deep reinforcement learning in a classic grid-maze game: **Pac-Man**. The task is deceptively simple—collect dots, avoid enemies—but it couples **long-horizon planning**, **partial observability**, and **tight navigation constraints** imposed by walls and corridors. Vanilla single-frame DQN often struggles here because a single observation cannot reveal motion (e.g., whether an enemy is approaching) or short-term trends (e.g., whether Pac-Man is reversing). To address this, we build a **frame-stacked DQN** agent that augments each observation with a short history window, giving the network access to implicit velocity and intent without the complexity of recurrent models.

Beyond the network input, reward design is pivotal. Sparse “win/lose/eat-dot” signals alone lead to slow or unstable learning, especially in mazes with dead ends. We adopt **potential-based reward shaping** that is *policy-invariant*: a small shaping term encourages moving **toward the nearest dot** and **away from the nearest enemy**, where distances are computed on the **walkable grid via BFS** (not Manhattan), so the agent is never misled by walls. This is combined with clear terminal rewards (clear-all, death, timeout), a light living cost, and gentle penalties for wall bumps and instantaneous reversals to reduce oscillation.

Learning stability is further improved with **Double DQN** targets and **Prioritized Experience Replay (PER)**, so the agent focuses on transitions with large TD error while correcting sampling bias via importance weights. We also use a **target network** for bootstrapping, ε-greedy exploration with decay, and optional gradient clipping.

For clarity and reproducibility, the environment features **fixed spawns** (Pac-Man near the center, enemies near the four corners), a compact **state encoder** (flattened grid + direction one-hot), and a **renderer** with an optional **Q-value heatmap** for diagnostics. The codebase includes training, testing, checkpointing, and replay-memory persistence to resume long runs. Taken together, these design choices make the project a practical, self-contained case study of modern DQN techniques on a nontrivial control problem.


## 2. Project Structure

A quick map of the repository and what each piece does.

```
.
├── agent.py              # DQN Agent (policy/target nets, epsilon policy, frame stacking, step/reset, save/load)
├── config.py             # All hyperparameters & constants (GRID_SIZE, INPUT_SIZE, HIDDEN_SIZE, LR, GAMMA, etc.)
├── environment.py        # Maze world: state encoding, reward, movement/step(), fixed spawns, helpers (BFS distance)
├── learning.py           # Model & learning utilities: DQN, train_dqn (Double DQN + PER), select_action helper
├── replayMemory.py       # Prioritized replay buffer (push/sample/update_priorities)
├── renderer.py           # Pygame renderer & heatmap diagnostics
├── train.py              # Training entry point (loop, logging, checkpointing, memory save)
├── test_pacman.py   # Greedy evaluation with rendering (loads a checkpoint)
└── checkpoints/          # Saved artifacts
    ├── latest_model.pth
    ├── memory.pkl
    └── meta.json
```

### Module-by-module

* **`config.py`**

  * Central place for constants (e.g., `K_FRAMES`, `INPUT_SIZE`, `OUTPUT_SIZE`, `HIDDEN_SIZE`, `BATCH_SIZE`, `EPSILON_*`, `TARGET_UPDATE_FREQ`, `CHECKPOINT_DIR`).
  * Changing stack size or maze size should only require updating config (and the derived `INPUT_SIZE`).

* **`environment.py`**

  * **Maze & spawns:** parses `MAZE`, computes `grid_w/h`, pixel dims, fixed player/enemy spawns.
  * **Step API:** `next_state, reward, done = env.step(action)`.
  * **State encoder:** `get_state() -> torch.FloatTensor[1, W*H + 4]` (flattened grid + direction one-hot).
  * **Reward:** potential-based shaping on BFS grid distance (toward dots, away from enemies) + terminal/living terms.
  * **Helpers:** `_is_walkable`, `_grid_distance_to_set` (multi-source BFS), coordinate conversions.

* **`agent.py`**

  * Wraps the RL logic around the environment.
  * **Networks:** `policy_net`, `target_net` (both `DQN` from `learning.py`).
  * **Frame stacking:** maintains a deque of the last `K_FRAMES` single-frame states; exposes `_stacked()`.
  * **Acting:** `select_action(state=None)` (ε-greedy) and `step(env, action)` → returns `(stacked_t, a, r, stacked_t1, done)`.
  * **Training hooks:** `optimize_model()`, `update_target()`, optional soft update if used.
  * **I/O:** `save(model_path, memory_path)`, optional `try_load_memory/save_memory` if enabled.
  * **Diagnostics:** `compute_heatmap(env)` (batch inference + K-frame approximation for visualization).

* **`learning.py`**

  * **`DQN`**: MLP with LeakyReLU, sized for `INPUT_SIZE` (frame-stacked).
  * **`train_dqn`**: one training step (Double DQN target, PER importance weights, SmoothL1/Huber loss, optimizer step, priority update, optional grad clip).
  * **`select_action`**: stateless helper for ε-greedy evaluation.

* **`replayMemory.py`**

  * **`ReplayMemory`** with Prioritized Experience Replay:

    * `push(state, action, reward, next_state, done)`
    * `sample(batch_size, beta)` → `(batch, indices, weights)`
    * `update_priorities(indices, td_errors)`
  
* **`renderer.py`**

  * Pygame window management, grid/dots/enemies rendering.
  * `draw_heatmap()` converts a Q-value grid into a colored overlay.

* **`train.py`**

  * Orchestrates episodes: reset, step loop, epsilon schedule, target updates, logging.
  * Saves **model** (`latest_model.pth`), **meta** (`meta.json`), and optionally **replay memory** (`memory.pkl`) at intervals.

* **`test_pacman.py`**

  * Loads a checkpoint, runs greedy (ε=0) episodes with rendering.
  * Uses the same frame-stacked `Agent` interface (`reset_episode`, `select_action`, `step`).

### Data & shape conventions (quick reference)

* **Actions:** discrete {0: left, 1: right, 2: up, 3: down}.
* **Single-frame state:** `[1, BASE_FEAT_DIM]` where `BASE_FEAT_DIM = W*H + 4`.
* **Stacked state:** `[1, INPUT_SIZE]` where `INPUT_SIZE = BASE_FEAT_DIM * K_FRAMES`.
* **Replay item:** `(state, action, reward, next_state, done)` with `state/next_state` = stacked tensors.
* **Batches in training:**
  `states, next_states → [B, INPUT_SIZE]`, `actions/rewards/dones → [B]`.


## 3. Methods

This section explains the learning choices in theory, but all statements reflect what the current implementation actually does.

### 3.1 Deep Q-Network (DQN)

We learn an action-value function $Q_\theta(s,a)$ with a feed-forward MLP (LeakyReLU non-linearities). Given a **discrete** action set $\mathcal{A}=\{\leftarrow,\rightarrow,\uparrow,\downarrow\}$, the network outputs a 4-vector of Q-values per input state. The learning objective is the standard **temporal-difference** regression toward a bootstrapped target:

$$
\mathcal{L}(\theta)=\mathbb{E}\,\big[\; \mathrm{Huber}\big(Q_\theta(s_t,a_t)-y_t\big)\;\big],
$$

with the **Huber** (Smooth L1).

We use **Double DQN** targets to reduce over-estimation bias:

$$
a^{*} = \operatorname*{arg\,max}_{a} \, Q_{\theta}(s_{t+1}, a), \qquad
y_t =
\begin{cases}
r_t, & \text{if done},\\[2pt]
r_t + \gamma\, Q_{\bar{\theta}}(s_{t+1}, a^{*}), & \text{otherwise}.
\end{cases}
$$


where $Q_{\bar\theta}$ is a **target network** (a delayed copy of $Q_\theta$). In words: the **online** network chooses the maximizing action at the next state; the **target** network evaluates it. This decoupling is enough to curb the classic maximization bias of vanilla DQN.

### 3.2 Frame Stacking (Partial Observability → Short-Horizon Memory)

A single Pac-Man frame does not reveal velocities or short-term trends (e.g., *who is approaching?* *did we just reverse?*). To mitigate partial observability without the complexity of RNNs, we feed the network a **stack** of the last $K$ observations:

$$
S_t = [s_{t-K+1},\,\dots,\,s_t] \in \mathbb{R}^{K\cdot d}.
$$

This “sliding window” supplies implicit **velocity/intent** cues through finite differences across frames. It typically stabilizes learning (fewer oscillations) and improves reaction quality near intersections. In replay, each transition is $(S_t, a_t, r_t, S_{t+1}, \text{done})$; if `done=True`, $S_{t+1}$ is terminal and we do **not** bootstrap.

### 3.3 State Encoding (What the Network Sees)

Each single frame $s_t$ is a compact vector made of:

* a **flattened occupancy grid** of the maze (walls, dots, enemies, Pac-Man) and
* a **one-hot** for Pac-Man’s current movement direction.

The stacked input is simply the concatenation of $K$ such frames. This balances **local geometric cues** (walls/corridors) with **agent dynamics** (direction one-hot) while staying light enough for fast training.

### 3.4 Reward Design with Potential-Based Shaping

Sparse signals (“win/lose/eat-dot”) alone are slow to learn in mazes. We therefore add **potential-based** shaping, which is *policy-invariant*: it does not change the set of optimal policies when written as

$$
F(s,a,s')=\gamma \Phi(s')-\Phi(s).
$$

Our potentials are:

* **Toward dots:** $\Phi_{\text{dot}}(s)=-d_{\text{dot}}(s)$
* **Away from enemies:** $\Phi_{\text{enm}}(s)=+d_{\text{enm}}(s)$

Crucially, distances $d(\cdot)$ are computed on the **walkable grid via BFS**, not Manhattan distance, so the agent is never “attracted through walls.” The final reward is:

* **Terminal:** large positive for clearing all dots; large negative for death; small negative for timeout/cap.
* **Instantaneous:** small living cost; positive for eating a dot; small penalties for **wall bumps** (no movement) and **immediate reversals** (to suppress oscillation).
* **Shaping:** weighted sum of the two potentials above with the same $\gamma$ as learning.

Intuition: the base events set the **task**, the shaping acts as a **gentle compass** (not a new objective).

### 3.5 Prioritized Experience Replay (PER)

Uniform replay wastes updates on uninformative transitions. We sample transitions with probability

$$
P(i)=\frac{p_i^\alpha}{\sum_j p_j^\alpha},\quad p_i = |\delta_i|+\varepsilon,
$$

where $\delta_i$ is the current **TD error** of sample $i$. The exponent $\alpha$ controls prioritization strength; a small $\varepsilon$ ensures every item remains samplable.

To correct the induced bias we use **importance sampling (IS)** weights:

$$
w_i=\Big(\tfrac{1}{N}\cdot\tfrac{1}{P(i)}\Big)^\beta \; / \; \max_j w_j,
$$

which anneal with $\beta \rightarrow 1$ over training. The loss is a **weighted Huber** average $\mathbb{E}[w_i \cdot \mathrm{Huber}(\delta_i)]$. After each update, we refresh priorities with new $|\delta_i|$.

### 3.6 Exploration, Targets, and Stability

* **$\varepsilon$-greedy** with decay: start exploratory, then gradually exploit as $Q$ stabilizes.
* **Target network** updates: periodical hard copy (or optional soft updates) to keep bootstrapping stable.
* **Optimization details**: Adam with a small weight decay; optional gradient clipping; `no_grad` around targets to avoid graph pollution; `eval()`/`train()` toggles where needed to keep deterministic targets.
* **Episode caps as terminals**: time/step caps are treated as `done=True` in replay so the learner sees **terminal samples** regularly, anchoring value scales.

---


## 4. Training Setup

This summarizes *how training actually runs in this repo*—the schedules, caps, buffers, and what the numbers mean. Values shown in parentheses reflect current defaults in `config.py` or module defaults.

### 4.1 Episodes, Caps, and Loop

* **Episode caps.** Each episode ends on **terminal** (death or all dots cleared) or on caps:
  **step cap** (`MAX_STEPS_PER_EPISODE=1000`) or **time cap** (`MAX_EPISODE_TIME=30s`).
  Caps are treated as *terminal* in replay so the learner regularly sees `done=True` transitions.
* **Act–learn cadence.** On every environment step we store one stacked transition and do one gradient update.

### 4.2 State, Batches, and Stacking

* **State.** The DQN sees **K-frame stacked** observations $S_t=[s_{t-K+1},\dots,s_t]$.
  Single-frame features are (flattened grid + direction one-hot).
  Effective input width is `INPUT_SIZE = BASE_FEAT_DIM * K_FRAMES`.
* **Batching.** Mini-batch TD learning with batch size **B** (`BATCH_SIZE=50`).

### 4.3 Optimization & Targets

* **Loss.** Smooth L1 (**Huber**) on the TD error.
* **Discount.** $\gamma = 0.99$ (`GAMMA=0.99`).
* **Optimizer.** Adam with learning rate **3e-4** (`LR=0.0003`).
  (A small weight decay may be applied in the agent to stabilize fitting.)
* **Double DQN target.** Action selection by the **online** net, evaluation by the **target** net:
$$
a^{*} = \operatorname*{arg\,max}_{a} Q_{\text{online}}(S_{t+1}, a), \qquad
y_t =
\begin{cases}
r_t, & \text{if terminal},\\[2pt]
r_t + \gamma\, Q_{\text{target}}(S_{t+1}, a^{*}), & \text{otherwise}.
\end{cases}
$$

* **Target updates.** Hard copy every **N episodes** (`TARGET_UPDATE_FREQ=10`).

### 4.4 Exploration (ε-greedy)

* Start fully exploratory `ε_start=1.0`, then **decay per episode** by `ε_decay=0.9995` down to
  `ε_min=0.05`. Greedy evaluation uses `ε=0.0`.

### 4.5 Replay Buffer (PER)

* **Buffer.** Capacity `MEMORY_CAPACITY=10000`, ring buffer on CPU tensors.
* **Prioritization.** **PER** with exponent $\alpha$ (default `0.6` in the buffer).
  New samples receive current **max priority** so they’re sampled at least once.
* **Sampling.** Draw a batch with **importance sampling** weights $w_i$ using $\beta$.
* **Terminals in batch.** The buffer can enforce a minimum number of terminal samples per batch
  (e.g., `min_terminal_samples=10`) so targets get regularly “anchored” by `done=True` cases.
* **Priority updates.** After backprop, priorities are updated from fresh $|\delta|$.

### 4.6 Reward Scheme (for context)

* **Terminal:** big positive (clear all), big negative (death), small negative (timeout/cap).
* **Per-step:** light living cost; + for dot; small penalties for wall-bump and instant reversal.
* **Shaping:** potential-based terms w\.r.t. **BFS shortest paths** (toward dots, away from enemies) with the same $\gamma$.
  Shaping nudges behavior without changing optimal policies.

### 4.7 Checkpointing & Resume

* **Artifacts.**

  * `checkpoints/latest_model.pth` — current policy weights
  * `checkpoints/meta.json` — episode/epsilon metadata
  * `checkpoints/memory.pkl` — pickled replay buffer (optional, for long runs)
* **Snapshots.** A full snapshot is written periodically (e.g., every 100 episodes).
* **Resume.** On startup you can load the model and (optionally) the replay memory to continue training without a cold buffer.

### 4.8 Monitoring & Diagnostics

* **Console logs** periodically print: moving **loss**, **mean Q**, and **#terminal samples** in the last batch.
* **Heatmap (optional).** A diagnostic Q-value heatmap renders the spatial preference of the current policy over the maze; for visualization it approximates stacking by repeating the single frame K times.

### 4.9 Environment & Reproducibility

* **Engine.** Python, PyTorch, NumPy (2.0-compatible), and Pygame for rendering.
* **Device.** GPU if available; otherwise CPU runs (slower but functional).
* **Seeds.** For strict reproducibility, fix Python/NumPy/PyTorch seeds and disable nondeterministic kernels; the project runs deterministically enough for comparison even without strict seeding.

### 4.10 Typical Commands

```bash
# Test (greedy[train.py](train.py) play with rendering)
python test_pacman.py --model checkpoints/latest_model.pth --fps 30
```

## 5. Results (Preliminary)

Due to hardware constraints, we have not run extended training or large-scale evaluations yet. The implementation has been validated to **run end-to-end** (training loop, PER sampling, Double DQN targets, frame stacking, rendering/heatmap, checkpoint & replay-memory persistence). Early smoke tests show stable loss reduction, but quantitative results are intentionally omitted until longer runs are feasible.

**When resources are available, we plan to report:**

* **Learning curves:** episode return, loss, mean/max Q.
* **Ablations:** K=1 vs K>1 (frame stacking), with/without shaping, with/without PER.
* **Qualitative behavior:** survival time, collision rate, wall-bump rate, heatmap snapshots.
* **Reproducibility:** seed-averaged curves (≥3 seeds), same configs.
