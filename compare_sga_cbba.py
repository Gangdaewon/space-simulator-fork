#!/usr/bin/env python3
"""
Standalone SGA vs CBBA comparison script.
No pygame dependency. Replicates seed-4 initial conditions and runs both algorithms.
"""
import random
import math
import numpy as np
from copy import deepcopy

# ──────────────────────────── Config ────────────────────────────
SEED = 4
N_AGENTS = 10
N_TASKS = 50
MAX_TASKS = 5
LAMBDA = 0.999
MAX_SPEED = 3.25
WORK_RATE = 20.0

# ──────────────── Position / amount generation ──────────────────
def gen_positions(n, xmin, xmax, ymin, ymax, seed):
    random.seed(seed)
    return [(random.randint(xmin, xmax), random.randint(ymin, ymax)) for _ in range(n)]

agent_positions = gen_positions(N_AGENTS, 0, 1400, 0, 1000, seed=SEED)
task_positions  = gen_positions(N_TASKS, 40, 1300, 40, 700, seed=SEED + 1000)

# task amounts: random.uniform after position generation consumed the seed state
# We need to replicate the exact state. After gen_positions for tasks, the
# random state is at seed=1004 + 100 randint calls. Just set seed to get amounts:
random.seed(SEED + 1000)
_ = [(random.randint(40, 1300), random.randint(40, 700)) for _ in range(N_TASKS)]  # consume same calls
task_amounts = [random.uniform(6.0, 60.0) for _ in range(N_TASKS)]

# ──────────────────────── Helper functions ──────────────────────
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def score_along_path(agent_pos, path_task_ids, task_pos, task_amt):
    """Compute S^{p_i} per CBBA paper Eqn (11)."""
    cur = agent_pos
    reward = 0.0
    cum_dist = 0.0
    for tid in path_task_ids:
        nxt = task_pos[tid]
        cum_dist += dist(cur, nxt)
        reward += LAMBDA ** (cum_dist / MAX_SPEED + task_amt[tid] / WORK_RATE)
        cur = nxt
    return reward

def best_insertion(agent_pos, current_path, new_tid, task_pos, task_amt):
    """Find the best insertion idx and marginal gain for new_tid into current_path."""
    S_p = score_along_path(agent_pos, current_path, task_pos, task_amt)
    best_idx = 0
    best_gain = float('-inf')
    for idx in range(len(current_path) + 1):
        alt = current_path[:idx] + [new_tid] + current_path[idx:]
        gain = score_along_path(agent_pos, alt, task_pos, task_amt) - S_p
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx, best_gain

# ════════════════════════════════════════════════════════════════
#                        SGA ALGORITHM
# ════════════════════════════════════════════════════════════════
def run_sga():
    bundles = {i: [] for i in range(N_AGENTS)}
    paths   = {i: [] for i in range(N_AGENTS)}
    winning_bids   = {}   # task_id -> bid
    winning_agents = {}   # task_id -> agent_id

    remaining = list(range(N_TASKS))
    N_min = min(N_TASKS, N_AGENTS * MAX_TASKS)

    for rnd in range(N_min):
        if not remaining:
            break

        best_gain = float('-inf')
        best_agent = best_task = best_idx = None

        for a in range(N_AGENTS):
            if len(bundles[a]) >= MAX_TASKS:
                continue

            # Compute bids for remaining tasks not in this agent's path
            agent_bid = {}
            agent_ins = {}
            for tid in remaining:
                if tid in paths[a]:
                    continue
                idx, gain = best_insertion(agent_positions[a], paths[a], tid,
                                           task_positions, task_amounts)
                agent_bid[tid] = gain
                agent_ins[tid] = idx

            # Apply winning bid filter (get_best_task logic)
            for tid, wb in winning_bids.items():
                if tid in agent_bid and wb > agent_bid[tid]:
                    agent_bid[tid] = float('-inf')

            if not agent_bid:
                continue
            best_tid = max(agent_bid, key=agent_bid.get)
            if agent_bid[best_tid] <= float('-inf'):
                continue

            if agent_bid[best_tid] > best_gain:
                best_gain = agent_bid[best_tid]
                best_agent = a
                best_task = best_tid
                best_idx = agent_ins[best_tid]

        if best_agent is None:
            print(f"[SGA] Round {rnd+1}: No agent found, breaking")
            break

        # Assign
        old_winner = winning_agents.get(best_task)
        winning_bids[best_task] = best_gain
        winning_agents[best_task] = best_agent
        bundles[best_agent].append(best_task)
        paths[best_agent].insert(best_idx, best_task)

        print(f"[SGA] Round {rnd+1:>2d}: Agent {best_agent} → Task {best_task:>2d} "
              f"(gain={best_gain:.6f}) | Bundle: {bundles[best_agent]}")

        # Outbid truncation
        if old_winner is not None and old_winner != best_agent:
            b = bundles[old_winner]
            n_bar = len(b)
            for idx, tid in enumerate(b):
                if winning_agents.get(tid) != old_winner:
                    n_bar = idx
                    break
            removed = set(b[n_bar:])
            for tid in list(b[n_bar+1:]):
                winning_bids[tid] = float('-inf')
                winning_agents[tid] = None
            bundles[old_winner] = b[:n_bar]
            paths[old_winner] = [t for t in paths[old_winner] if t not in removed]
            if removed:
                print(f"[SGA]   └─ OUTBID Agent {old_winner}: removed {removed}")

        remaining = [t for t in remaining if t != best_task]

    return bundles, paths

# ════════════════════════════════════════════════════════════════
#                        CBBA ALGORITHM
# ════════════════════════════════════════════════════════════════
def run_cbba():
    bundles = {i: [] for i in range(N_AGENTS)}
    paths   = {i: [] for i in range(N_AGENTS)}
    y = {i: {} for i in range(N_AGENTS)}   # winning bid per agent
    z = {i: {} for i in range(N_AGENTS)}   # winning agent per agent

    all_tasks = list(range(N_TASKS))
    max_ticks = 50
    converged = [False] * N_AGENTS

    for tick in range(1, max_ticks + 1):
        if all(converged):
            break

        # ── Phase 1: BUILD BUNDLE for each agent ──
        for a in range(N_AGENTS):
            if converged[a]:
                continue
            while len(bundles[a]) < MAX_TASKS:
                # Compute bids for tasks not in path
                agent_bid = {}
                agent_ins = {}
                for tid in all_tasks:
                    if tid in paths[a]:
                        continue
                    idx, gain = best_insertion(agent_positions[a], paths[a], tid,
                                               task_positions, task_amounts)
                    agent_bid[tid] = gain
                    agent_ins[tid] = idx

                # Apply winning bid filter
                for tid, wb in y[a].items():
                    if tid in agent_bid and wb > agent_bid[tid]:
                        agent_bid[tid] = float('-inf')

                if not agent_bid:
                    break
                best_tid = max(agent_bid, key=agent_bid.get)
                if agent_bid[best_tid] <= float('-inf'):
                    break

                bundles[a].append(best_tid)
                paths[a].insert(agent_ins[best_tid], best_tid)
                y[a][best_tid] = agent_bid[best_tid]
                z[a][best_tid] = a

                if len(bundles[a]) > MAX_TASKS:
                    last = bundles[a][-1]
                    y[a][last] = 0
                    z[a][last] = None
                    bundles[a] = bundles[a][:MAX_TASKS]
                    paths[a] = [t for t in paths[a] if t != last]

        if tick == 1:
            print(f"\n[CBBA] === Tick {tick} BUILD BUNDLE results ===")
            for a in range(N_AGENTS):
                print(f"  Agent {a}: Bundle={bundles[a]} Path={paths[a]}")

        # ── Phase 2: CONSENSUS (fully connected: each agent sees all others) ──
        for a in range(N_AGENTS):
            if converged[a]:
                continue
            for tid in all_tasks:
                for k in range(N_AGENTS):
                    if k == a:
                        continue
                    if y[k].get(tid) is None or y[a].get(tid) is None:
                        continue

                    z_k_j = z[k].get(tid)
                    y_k_j = y[k].get(tid, 0)
                    z_i_j = z[a].get(tid)
                    y_i_j = y[a].get(tid, 0)

                    # Simplified consensus: higher bid wins
                    if z_k_j == k:  # k claims to be winner
                        if z_i_j == a:  # I also claim to be winner
                            if y_k_j > y_i_j:
                                y[a][tid] = y_k_j
                                z[a][tid] = k
                        elif z_i_j == k:
                            y[a][tid] = y_k_j
                            z[a][tid] = k
                        elif z_i_j is None:
                            y[a][tid] = y_k_j
                            z[a][tid] = k
                        else:
                            if y_k_j > y_i_j:
                                y[a][tid] = y_k_j
                                z[a][tid] = k

        # ── Update bundles ──
        any_changed = False
        for a in range(N_AGENTS):
            if converged[a]:
                continue
            # Find first outbid point
            n_bar = len(bundles[a])
            for idx, tid in enumerate(bundles[a]):
                if z[a].get(tid) != a:
                    n_bar = idx
                    break
            if n_bar < len(bundles[a]):
                removed = set(bundles[a][n_bar:])
                for tid in list(bundles[a][n_bar+1:]):
                    y[a][tid] = float('-inf')
                    z[a][tid] = None
                new_bundle = bundles[a][:n_bar]
                new_path = [t for t in paths[a] if t not in removed]
                if tick <= 3 or not converged[a]:
                    print(f"[CBBA] Tick {tick}: Agent {a} TRUNCATED: {bundles[a]} → {new_bundle} (removed {removed})")
                bundles[a] = new_bundle
                paths[a] = new_path
                any_changed = True
                converged[a] = False
            else:
                # Bundle unchanged - check if fully built
                if len(bundles[a]) == MAX_TASKS:
                    if not converged[a]:
                        print(f"[CBBA_CONVERGED] Tick {tick}: Agent {a} Bundle={bundles[a]} Path={paths[a]}")
                    converged[a] = True

        if not any_changed and all(len(bundles[a]) == MAX_TASKS for a in range(N_AGENTS)):
            break

    return bundles, paths

# ════════════════════════════════════════════════════════════════
#                           MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Agent positions (seed=4):")
    for i, p in enumerate(agent_positions):
        print(f"  Agent {i}: {p}")
    print(f"\nTask positions (first 5): {task_positions[:5]}")
    print(f"Task amounts (first 5): {[f'{a:.1f}' for a in task_amounts[:5]]}")
    print("=" * 60)

    print("\n\n{'='*60}")
    print("RUNNING SGA...")
    print("=" * 60)
    sga_bundles, sga_paths = run_sga()

    print(f"\n\n{'='*60}")
    print("RUNNING CBBA...")
    print("=" * 60)
    cbba_bundles, cbba_paths = run_cbba()

    # ── Comparison ──
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Agent':>6} | {'SGA Bundle':<30} | {'CBBA Bundle':<30} | Match")
    print("-" * 80)
    for a in range(N_AGENTS):
        sb = sga_bundles[a]
        cb = cbba_bundles[a]
        match = "✅" if sb == cb else "❌"
        print(f"{a:>6} | {str(sb):<30} | {str(cb):<30} | {match}")

    # Show which tasks are different
    print(f"\n{'='*80}")
    print("TASK OWNERSHIP DIFFERENCES")
    print("=" * 80)
    sga_ownership = {}
    cbba_ownership = {}
    for a in range(N_AGENTS):
        for t in sga_bundles[a]:
            sga_ownership[t] = a
        for t in cbba_bundles[a]:
            cbba_ownership[t] = a

    diffs = []
    for t in range(N_TASKS):
        so = sga_ownership.get(t, "NONE")
        co = cbba_ownership.get(t, "NONE")
        if so != co:
            diffs.append((t, so, co))
            print(f"  Task {t:>2d}: SGA→Agent {so}, CBBA→Agent {co}")

    if not diffs:
        print("  No differences!")
    else:
        print(f"\n  Total different tasks: {len(diffs)}")
