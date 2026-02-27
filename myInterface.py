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

evaluate_btn = st.sidebar.button("🚀 Evaluate Population")
multi_btn = st.sidebar.button("📊 Run Multiple Populations")

def generate_population(func,N, D):
    if "F1" in func:
        return np.random.uniform(-100, 100, D)
    elif "F2" in func:
        return np.random.uniform(-10, 10, D)
    elif "F5" in func:
        return np.random.uniform(-30, 30, D)
    elif "F7" in func:
        return np.random.uniform(-128, 128, D)
    elif "F9" in func:
        return np.random.uniform(-5.12, 5.12, D)
    elif "F11" in func:
        return np.random.uniform(-600, 600, D)



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

    # plot population (first two variables only)
    ax.scatter(population[:,0], population[:,1],
               color='red', label="Population")

    # highlight best solution
    ax.scatter(population[best_idx,0],
               population[best_idx,1],
               color='blue', s=100, label="Best Solution")

    ax.set_title("Contour + Population")
    ax.legend()
    st.pyplot(fig)


# ---------------- MULTIPLE RUNS ----------------
if multi_btn:

    R = st.sidebar.slider("Number of Runs (R)", 5, 100, 20)
    f = functions[function_name]

    best_list = []
    worst_list = []

    # ---------------- RUNS LOOP ----------------
    for i in range(R):

        pop = generate_population(function_name, N, D)
        fitness = np.array([f(ind) for ind in pop])

        Bi = np.min(fitness)
        Wi = np.max(fitness)

        best_list.append(Bi)
        worst_list.append(Wi)

    # ---------------- STATISTICS ----------------
    BEST = np.min(best_list)
    WORST = np.max(worst_list)
    AVG = np.mean(best_list)
    STD = np.std(best_list, ddof=1)

    st.subheader("📊 Multiple Runs Performance")

    col1, col2 = st.columns(2)
    col1.metric("Global Best", round(float(BEST), 6))
    col2.metric("Global Worst", round(float(WORST), 6))

    col3, col4 = st.columns(2)
    col3.metric("AVG (Mean Best)", round(float(AVG), 6))
    col4.metric("STD (Stability)", round(float(STD), 6))

    # ---------------- EVOLUTION PLOT ----------------
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

    # ---------------- SCATTER PLOT OF RUNS ----------------
    st.subheader("📍 Scatter Plot of Best Fitness per Run")

    fig2, ax2 = plt.subplots()
    ax2.scatter(runs, best_list)
    ax2.set_xlabel("Run Number")
    ax2.set_ylabel("Best Fitness (Bi)")
    ax2.set_title("Scatter of Best Fitness Across Runs")
    ax2.grid(True)

    st.pyplot(fig2)

    # ---------------- INTERPRETATION ----------------
    if STD < 1 and AVG < 1:
        st.success("Ideal balance: stable and high-quality solutions.")
    elif STD < 1 and AVG > 1:
        st.warning("Low diversity: possible premature convergence.")
    else:
        st.info("High exploration behavior.")