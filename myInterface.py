import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG PAGE (UI MODERNE) ----------------
st.set_page_config(
    page_title="META TP - Optimization Benchmark",
    page_icon="📊",
    layout="wide"
)

# ---------------- STYLE CSS MODERNE ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #00F5A0;}
.stButton>button {
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BENCHMARK FUNCTIONS ----------------
def F1(x):  # Sphere
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F5(x):  # Rosenbrock
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def F7(x):
    i = np.arange(1, len(x)+1)
    return np.sum(i * (x**4)) + np.random.rand()

def F9(x):  # Rastrigin
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

def F11(x):  # Griewank
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return 1 + sum_term - prod_term

functions = {
    "F1 - Sphere": F1,
    "F2": F2,
    "F5 - Rosenbrock": F5,
    "F7": F7,
    "F9 - Rastrigin": F9,
    "F11 - Griewank": F11
}

# ---------------- HEADER ----------------
st.title("📊 META.H - Benchmark Functions Dashboard")
st.markdown("### Interface pour tester les fonctions d'optimisation. ")

## ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Parameters")

function_name = st.sidebar.selectbox("Choose Function", list(functions.keys()))
D = st.sidebar.slider("Dimension (D)", 2, 50, 2)
N = st.sidebar.slider("Population Size (N)", 5, 200, 30)

init_mode = st.sidebar.radio("Initialization Mode",
                              ["Random Population", "Load CSV"])

evaluate_btn = st.sidebar.button("🚀 Evaluate Population", key="evaluate")
multi_btn = st.sidebar.button("📊 Run Multiple Populations", key="multi")
run_pso_btn = st.sidebar.button("🚀 Run PSO Optimization", key="pso")


def generate_population(func,N, D):
    if "F1" in func:
        return np.random.uniform(-100, 100, (N, D))
    elif "F2" in func:
        return np.random.uniform(-10, 10, (N, D))
    elif "F5" in func:
        return np.random.uniform(-30, 30, (N, D))
    elif "F7" in func:
        return np.random.uniform(-128, 128, (N, D))
    elif "F9" in func:
        return np.random.uniform(-5.12, 5.12, (N, D))
    elif "F11" in func:
        return np.random.uniform(-600, 600, (N, D))

# ---------------- INITIALIZATION ----------------
population = None

if init_mode == "Random Population":
    population = generate_population(function_name, N, D)
else:
    uploaded_file = st.file_uploader("Upload CSV Population", type="csv")
    if uploaded_file:
        population = pd.read_csv(uploaded_file, header=None, sep=';').values.astype(float)
        N, D = population.shape

# ---------------- SINGLE POPULATION EVALUATION ----------------
if evaluate_btn and population is not None:
    f = functions[function_name]
    fitness = np.array([f(ind) for ind in population])
    best_idx = np.argmin(fitness)
    worst_idx = np.argmax(fitness)
    best_value = fitness[best_idx]
    worst_value = fitness[worst_idx]

    st.subheader("📈 Population Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Best Fitness", round(float(best_value), 6))
    col2.metric("Worst Fitness", round(float(worst_value), 6))

    # ---------------- CONTOUR + SCATTER ----------------
    st.subheader("📊 Population Visualization (x1, x2)")
    x_min, x_max = -5, 5
    X = np.linspace(x_min, x_max, 200)
    Y = np.linspace(x_min, x_max, 200)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X)):
            point = np.array([X[i,j], Y[i,j]] + [0]*(D-2))
            Z[i,j] = f(point)

    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, 30)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.scatter(population[:,0], population[:,1], color='red', label="Population")
    ax.scatter(population[best_idx,0], population[best_idx,1], color='blue', s=100, label="Best Solution")
    ax.set_title("Contour + Population")
    ax.legend()
    st.pyplot(fig)

# ---------------- MULTIPLE RUNS ----------------
if multi_btn:
    R = st.sidebar.slider("Number of Runs (R)", 5, 100, 20)
    f = functions[function_name]
    if population is None and init_mode == "Load CSV":
        st.error("Please upload a CSV population first.")
        st.stop()
    best_list, worst_list = [], []
    for i in range(R):
        if init_mode == "Load CSV" and population is not None:
            pop = population
        else:
            pop = generate_population(function_name, N, D)
        fitness = np.array([f(ind) for ind in pop])
        best_list.append(np.min(fitness))
        worst_list.append(np.max(fitness))
    BEST, WORST, AVG, STD = np.min(best_list), np.max(worst_list), np.mean(best_list), np.std(best_list, ddof=1)

    st.subheader("📊 Multiple Runs Performance")
    col1, col2 = st.columns(2)
    col1.metric("Global Best", round(float(BEST), 6))
    col2.metric("Global Worst", round(float(WORST), 6))
    col3, col4 = st.columns(2)
    col3.metric("AVG (Mean Best)", round(float(AVG), 6))
    col4.metric("STD (Stability)", round(float(STD), 6))

    if init_mode == "Load CSV":
        st.info("CSV mode detected: runs evaluate the SAME population (deterministic results).")
    else:
        st.info("Random mode: each run generates a new random population.")

    st.subheader("📈 Evolution of Best Fitness Across Runs")
    runs = np.arange(1, R + 1)
    fig1, ax1 = plt.subplots()
    ax1.plot(runs, best_list, marker='o', label="Best (Bi)")
    ax1.plot(runs, worst_list, linestyle='--', label="Worst (Wi)")
    ax1.set_xlabel("Run Number")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Best and Worst Fitness per Run")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("📍 Scatter Plot of Best Fitness per Run")
    fig2, ax2 = plt.subplots()
    ax2.scatter(runs, best_list)
    ax2.set_xlabel("Run Number")
    ax2.set_ylabel("Best Fitness (Bi)")
    ax2.set_title("Scatter of Best Fitness Across Runs")
    ax2.grid(True)
    st.pyplot(fig2)

# ---------------- PARTICLE SWARM OPTIMIZATION ----------------
def PSO(func, N=30, D=2, T=200, w=0.3, c1=1.4, c2=1.4):
    # Initialisation
    X = np.random.uniform(-100, 100, (N, D))  # positions
    V = np.random.uniform(-1, 1, (N, D))      # vitesses
    pbest = X.copy()
    pbest_val = np.array([func(x) for x in X])
    gbest = X[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    history = [gbest_val]
    trajectories = [X[0].copy()]  # suivre particule 0

    for t in range(T):
        for i in range(N):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            V[i] = (w*V[i] + c1*r1*(pbest[i] - X[i]) + c2*r2*(gbest - X[i]))
            X[i] = X[i] + V[i]

            val = func(X[i])
            if val < pbest_val[i]:
                pbest[i] = X[i]
                pbest_val[i] = val
            if val < gbest_val:
                gbest = X[i]
                gbest_val = val

        history.append(gbest_val)
        trajectories.append(X[0].copy())

    return gbest, gbest_val, history, np.array(trajectories), X

# ---------------- RUN PSO ----------------

if run_pso_btn:
    f = functions[function_name]
    gbest, gbest_val, history, traj, final_pop = PSO(f, N, D, T=200)

    st.subheader("🏆 Résultats PSO")
    st.metric("Best Fitness (gbest)", round(float(gbest_val), 6))
    st.metric("Stagnation", len(history) - np.argmin(history))  # itérations depuis le meilleur

    # Courbe de convergence
    st.subheader("📈 Courbe de convergence")
    fig1, ax1 = plt.subplots()
    ax1.plot(history, label="Best Fitness")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Convergence Curve")
    ax1.legend()
    st.pyplot(fig1)

    # Trajectoire de la première particule
    if D >= 2:
        st.subheader("🌀 Trajectoire de la première particule")
        fig2, ax2 = plt.subplots()
        ax2.plot(traj[:,0], traj[:,1], marker='o')
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_title("Trajectory of Particle 0")
        st.pyplot(fig2)

    # Comparaison population initiale vs finale
    if population is not None and D >= 2:
        st.subheader("🔍 Population Initiale vs Finale")
        fig3, ax3 = plt.subplots()
        ax3.scatter(population[:,0], population[:,1], color='red', label="Initial Population")
        ax3.scatter(final_pop[:,0], final_pop[:,1], color='green', label="Final Population")
        ax3.set_xlabel("X1")
        ax3.set_ylabel("X2")
        ax3.set_title("Population Evolution")
        ax3.legend()
        st.pyplot(fig3)
