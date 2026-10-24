"""
Energy Efficient Routing in Wireless Sensor Networks Using ML
-------------------------------------------------------------
Compares:
  1. LEACH   (baseline, random cluster-head selection)
  2. ML-WSN  (K-Means clustering + energy-aware cluster-head selection)

Run:  python wsn_routing.py
Out:  graphs + metrics saved in the results/ folder
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------- Configuration ----------------
N = 100                  # number of sensor nodes
AREA = 100               # field size (100 x 100 m)
BS = np.array([50, 175]) # base station position
E0 = 0.5                 # initial energy per node (Joules)
E_ELEC = 50e-9           # electronics energy (J/bit)
E_FS = 10e-12            # free-space amplifier (J/bit/m^2)
E_MP = 0.0013e-12        # multipath amplifier (J/bit/m^4)
E_DA = 5e-9              # data aggregation energy (J/bit/signal)
PKT = 4000               # data packet size (bits)
P = 0.05                 # desired fraction of cluster heads (5%)
MAX_ROUNDS = 5000
SEED = 42
D0 = np.sqrt(E_FS / E_MP)  # crossover distance

os.makedirs("results", exist_ok=True)


# ---------------- Radio energy model ----------------
def tx_energy(bits, d):
    d = np.asarray(d, dtype=float)
    return np.where(d < D0,
                    bits * (E_ELEC + E_FS * d ** 2),
                    bits * (E_ELEC + E_MP * d ** 4))


def rx_energy(bits):
    return bits * E_ELEC


# ---------------- Network creation ----------------
def create_network(seed=SEED):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, AREA, size=(N, 2))
    energy = np.full(N, E0)
    return pos, energy


# ---------------- Cluster head selection ----------------
def leach_select(alive_idx, round_no, last_ch_round, rng):
    """Classic LEACH threshold-based random CH selection."""
    period = int(1 / P)
    denom = 1 - P * (round_no % period)
    thr = P / denom
    chs = []
    for i in alive_idx:
        eligible = (round_no - last_ch_round[i]) >= period
        if eligible and rng.random() < thr:
            chs.append(i)
    if not chs:  # make sure at least one CH exists
        chs = [int(rng.choice(alive_idx))]
    return np.array(chs)


def ml_select(pos, energy, alive_idx, seed):
    """K-Means clusters + CH = best score inside each cluster.
    score = 0.6*residual energy + 0.2*closeness to BS + 0.2*closeness to centroid
    """
    k = max(1, int(round(P * len(alive_idx))))
    k = min(k, len(alive_idx))
    pts = pos[alive_idx]
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(pts)
    labels, centers = km.labels_, km.cluster_centers_

    d_bs_all = np.linalg.norm(pts - BS, axis=1)
    chs, member_ch = [], {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        nodes = alive_idx[idx]
        e_norm = energy[nodes] / E0
        d_bs = d_bs_all[idx] / (d_bs_all.max() + 1e-9)
        d_ct = np.linalg.norm(pts[idx] - centers[c], axis=1)
        d_ct = d_ct / (d_ct.max() + 1e-9)
        score = 0.6 * e_norm + 0.2 * (1 - d_bs) + 0.2 * (1 - d_ct)
        ch = nodes[np.argmax(score)]
        chs.append(ch)
        for n in nodes:
            member_ch[n] = ch
    return np.array(chs), member_ch


# ---------------- Simulation ----------------
def simulate(protocol, seed=SEED):
    pos, energy = create_network(seed)
    rng = np.random.default_rng(seed)
    last_ch_round = np.full(N, -1000)

    alive_hist, energy_hist, packets_hist = [], [], []
    first_dead = half_dead = last_dead = None
    total_packets = 0

    for r in range(MAX_ROUNDS):
        alive_idx = np.where(energy > 0)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            last_dead = r
            break
        if first_dead is None and n_alive < N:
            first_dead = r
        if half_dead is None and n_alive <= N // 2:
            half_dead = r

        # ---- choose cluster heads & membership ----
        if protocol == "LEACH":
            chs = leach_select(alive_idx, r, last_ch_round, rng)
            last_ch_round[chs] = r
            member_ch = {}
            for n in alive_idx:
                if n in chs:
                    continue
                d = np.linalg.norm(pos[chs] - pos[n], axis=1)
                member_ch[n] = chs[np.argmin(d)]
        else:
            chs, member_ch = ml_select(pos, energy, alive_idx, seed + r)

        # ---- energy consumption ----
        counts = {int(c): 0 for c in chs}
        for n, c in member_ch.items():
            if n == c:
                continue
            d = np.linalg.norm(pos[n] - pos[c])
            energy[n] -= float(tx_energy(PKT, d))
            energy[c] -= rx_energy(PKT)
            counts[int(c)] += 1

        for c in chs:
            energy[c] -= E_DA * PKT * (counts[int(c)] + 1)
            d_bs = np.linalg.norm(pos[c] - BS)
            energy[c] -= float(tx_energy(PKT, d_bs))
            total_packets += 1

        energy = np.maximum(energy, 0)
        alive_hist.append(int(np.sum(energy > 0)))
        energy_hist.append(float(energy.sum()))
        packets_hist.append(total_packets)

    if last_dead is None:
        last_dead = MAX_ROUNDS
    return dict(alive=alive_hist, energy=energy_hist, packets=packets_hist,
                first=first_dead, half=half_dead, last=last_dead)


# ---------------- Plots ----------------
def plot_results(res):
    colors = {"LEACH": "tab:red", "ML-WSN": "tab:green"}

    for key, title, ylabel, fname in [
        ("alive", "Alive Nodes per Round", "Alive nodes", "alive_nodes.png"),
        ("energy", "Total Residual Energy per Round", "Energy (J)", "residual_energy.png"),
        ("packets", "Packets Delivered to Base Station", "Packets", "packets_delivered.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for name, r in res.items():
            plt.plot(r[key], label=name, color=colors[name])
        plt.title(title); plt.xlabel("Round"); plt.ylabel(ylabel)
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"results/{fname}", dpi=150); plt.close()

    # lifetime bar chart
    labels = ["First node dead", "Half nodes dead", "Last node dead"]
    x = np.arange(3); w = 0.35
    plt.figure(figsize=(7, 4.5))
    for i, (name, r) in enumerate(res.items()):
        vals = [r["first"], r["half"], r["last"]]
        plt.bar(x + i * w, vals, w, label=name, color=colors[name])
    plt.xticks(x + w / 2, labels); plt.ylabel("Round")
    plt.title("Network Lifetime Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/lifetime_comparison.png", dpi=150); plt.close()


# ---------------- Network picture (clusters + cluster heads) ----------------
def plot_network(seed=SEED, save_path=None):
    """Draw nodes, clusters, cluster heads and base station for both protocols."""
    pos, energy = create_network(seed)
    rng = np.random.default_rng(seed)
    alive_idx = np.arange(N)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # LEACH
    chs = leach_select(alive_idx, 0, np.full(N, -1000), rng)
    leach_labels = {}
    for n in alive_idx:
        leach_labels[n] = n if n in chs else chs[np.argmin(np.linalg.norm(pos[chs] - pos[n], axis=1))]
    # ML
    ml_chs, ml_labels = ml_select(pos, energy, alive_idx, seed)

    for ax, title, ch_list, labels in [(axes[0], "LEACH (random cluster heads)", chs, leach_labels),
                                       (axes[1], "ML-WSN (K-Means + energy-aware)", ml_chs, ml_labels)]:
        cmap = plt.get_cmap("tab20")
        ch_ids = {int(c): i for i, c in enumerate(ch_list)}
        for n in alive_idx:
            c = int(labels[n])
            ax.scatter(*pos[n], color=cmap(ch_ids[c] % 20), s=25)
            if n != c:
                ax.plot([pos[n][0], pos[c][0]], [pos[n][1], pos[c][1]], color=cmap(ch_ids[c] % 20), alpha=0.3, lw=0.7)
        ax.scatter(pos[ch_list][:, 0], pos[ch_list][:, 1], marker="*", s=220, color="black", label="Cluster head", zorder=5)
        ax.scatter(*BS, marker="^", s=150, color="red", label="Base station", zorder=5)
        ax.set_xlim(-5, AREA + 5); ax.set_ylim(-5, BS[1] + 15)
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# ---------------- Multiple runs (different random layouts) ----------------
def run_multi(n_runs=10):
    """Run both protocols on n_runs random layouts; return mean/std of each metric."""
    out = {"LEACH": [], "ML-WSN": []}
    for i in range(n_runs):
        seed = SEED + i
        print(f"  run {i + 1}/{n_runs} (seed {seed})")
        for name, proto in [("LEACH", "LEACH"), ("ML-WSN", "ML")]:
            r = simulate(proto, seed=seed)
            out[name].append([r["first"], r["half"], r["last"], r["packets"][-1]])
    return {k: np.array(v) for k, v in out.items()}


def plot_multi(multi, save_path=None):
    colors = {"LEACH": "tab:red", "ML-WSN": "tab:green"}
    labels = ["First node dead", "Half nodes dead", "Last node dead"]
    x = np.arange(3); w = 0.35
    fig = plt.figure(figsize=(7.5, 4.8))
    for i, (name, arr) in enumerate(multi.items()):
        mean, std = arr[:, :3].mean(axis=0), arr[:, :3].std(axis=0)
        plt.bar(x + i * w, mean, w, yerr=std, capsize=4, label=name, color=colors[name])
    plt.xticks(x + w / 2, labels); plt.ylabel("Round")
    plt.title(f"Average Lifetime over {len(next(iter(multi.values())))} Random Layouts")
    plt.legend(); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def save_multi_summary(multi):
    with open("results/multi_run_summary.csv", "w") as f:
        f.write("protocol,first_dead_mean,first_dead_std,half_dead_mean,half_dead_std,last_dead_mean,last_dead_std,packets_mean\n")
        for name, arr in multi.items():
            m, s = arr.mean(axis=0), arr.std(axis=0)
            f.write(f"{name},{m[0]:.1f},{s[0]:.1f},{m[1]:.1f},{s[1]:.1f},{m[2]:.1f},{s[2]:.1f},{m[3]:.1f}\n")
            print(f"{name:<8} first dead {m[0]:.0f} +/- {s[0]:.0f} | half dead {m[1]:.0f} +/- {s[1]:.0f} | "
                  f"last dead {m[2]:.0f} +/- {s[2]:.0f} | packets {m[3]:.0f}")


def main():
    results = {}
    for name, proto in [("LEACH", "LEACH"), ("ML-WSN", "ML")]:
        print(f"Running {name} ...")
        results[name] = simulate(proto)

    print("\n=== Results (rounds) ===")
    print(f"{'Protocol':<10}{'First dead':>12}{'Half dead':>12}{'Last dead':>12}{'Packets':>10}")
    with open("results/summary.txt", "w") as f:
        f.write("protocol,first_dead,half_dead,last_dead,packets\n")
        for name, r in results.items():
            pk = r["packets"][-1] if r["packets"] else 0
            print(f"{name:<10}{r['first']:>12}{r['half']:>12}{r['last']:>12}{pk:>10}")
            f.write(f"{name},{r['first']},{r['half']},{r['last']},{pk}\n")

    plot_results(results)
    plot_network(save_path="results/network_clusters.png")
    print("\nGraphs saved in results/ folder.")

    n_runs = int(os.environ.get("WSN_RUNS", 10))
    print(f"\n=== Multi-run average ({n_runs} random layouts) ===")
    multi = run_multi(n_runs)
    save_multi_summary(multi)
    plot_multi(multi, save_path="results/multi_run_lifetime.png")
    print("\nAll results saved in results/ folder.")


if __name__ == "__main__":
    main()
