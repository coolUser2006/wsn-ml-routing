"""
Streamlit demo for Energy Efficient Routing in WSN Using ML
Run:  streamlit run app.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import wsn_routing as w

st.set_page_config(page_title="WSN Energy Efficient Routing", layout="wide")
st.title("Energy Efficient Routing in Wireless Sensor Networks Using ML")
st.caption("LEACH (random cluster heads) vs ML-WSN (K-Means + energy-aware cluster heads)")

st.sidebar.header("Settings")
n_nodes = st.sidebar.slider("Number of nodes", 30, 200, 100, 10)
e0 = st.sidebar.slider("Initial energy per node (J)", 0.1, 1.0, 0.5, 0.1)
p = st.sidebar.slider("Cluster head fraction", 0.02, 0.15, 0.05, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
bs_y = st.sidebar.slider("Base station distance (y, metres)", 100, 250, 175, 25)
run = st.sidebar.button("Run simulation", type="primary")

# apply settings to the simulation module
w.N, w.E0, w.P = n_nodes, e0, p
w.BS = w.np.array([50, bs_y])
w.SEED = int(seed)

st.subheader("Network layout (round 1)")
fig = w.plot_network(seed=int(seed))
st.pyplot(fig)
plt.close(fig)

if run:
    with st.spinner("Simulating both protocols..."):
        res = {"LEACH": w.simulate("LEACH", seed=int(seed)),
               "ML-WSN": w.simulate("ML", seed=int(seed))}

    st.subheader("Network lifetime (rounds)")
    table = {name: {"First node dead": r["first"], "Half nodes dead": r["half"],
                    "Last node dead": r["last"], "Packets to BS": r["packets"][-1]}
             for name, r in res.items()}
    st.table(table)

    colors = {"LEACH": "tab:red", "ML-WSN": "tab:green"}
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "alive", "Alive nodes per round"), (c2, "energy", "Total residual energy (J)")]:
        f = plt.figure(figsize=(6, 4))
        for name, r in res.items():
            plt.plot(r[key], label=name, color=colors[name])
        plt.title(title); plt.xlabel("Round"); plt.grid(alpha=0.3); plt.legend()
        col.pyplot(f)
        plt.close(f)
else:
    st.info("Set the parameters on the left and click **Run simulation**.")
