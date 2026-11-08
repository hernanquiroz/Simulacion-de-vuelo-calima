
"""Cálculo en el Aire · 4 simulaciones de Cálculo Diferencial e Integral aplicadas a la aviación.

Ejecutar:  streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

CREDITO = "Diseño Vibe Coding por Ing. Hernán Quiroz"
G = 9.80665
_trapz = getattr(np, "trapezoid", None) or np.trapz  # compatible con NumPy 1.x y 2.x
AZUL, AMBAR, VERDE, ROJO = "#38bdf8", "#fbbf24", "#34d399", "#f87171"

st.set_page_config(page_title="Cálculo en el Aire", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ utilidades
def fig_base(titulo, xt, yt, **kw):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15)),
        xaxis_title=xt, yaxis_title=yt, height=360,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.25), **kw,
    )
    return fig


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def integrar(metodo, f, a, b, n):
    h = (b - a) / n
    if metodo == "Riemann izquierda":
        return h * sum(f(a + i * h) for i in range(n))
    if metodo == "Riemann derecha":
        return h * sum(f(a + i * h) for i in range(1, n + 1))
    if metodo == "Punto medio":
        return h * sum(f(a + (i + 0.5) * h) for i in range(n))
    if metodo == "Trapecio":
        return h * ((f(a) + f(b)) / 2 + sum(f(a + i * h) for i in range(1, n)))
    s = f(a) + f(b) + sum(f(a + i * h) * (4 if i % 2 else 2) for i in range(1, n))
    return s * h / 3  # Simpson 1/3 (n par)


# ------------------------------------------------------------------ Sim 1
def sim_derivada():
    st.subheader("1 · Tasa de ascenso como derivada de la altitud")
    st.markdown(
        "Un avión asciende hacia su techo con **h(t) = H·(1 − e^(−t/τ))**. "
        "La tasa de ascenso (ROC) es **h′(t)**. Reduce Δt y observa cómo la secante se acerca a la tangente."
    )
    c1, c2, c3, c4 = st.columns(4)
    H = c1.slider("Techo asintótico H (ft)", 10000, 45000, 20000, 500)
    tau = c2.slider("Constante τ (min)", 5, 40, 18)
    t0 = c3.slider("Instante t₀ (min)", 0.0, 60.0, 10.0, 0.5)
    dt = c4.slider("Incremento Δt (min)", 0.05, 30.0, 10.0, 0.05)

    h = lambda t: H * (1 - np.exp(-t / tau))
    dh = lambda t: (H / tau) * np.exp(-t / tau)
    T = 70
    ts = np.linspace(0, T, 281)
    h0, m = h(t0), dh(t0)
    t1 = t0 + dt
    h1 = h(t1)
    ms = (h1 - h0) / dt
    tx = np.array([max(0, t0 - 20), min(T, t0 + 20)])

    f1 = fig_base("Altitud h(t)", "Tiempo (min)", "Altitud (ft)")
    f1.add_scatter(x=ts, y=h(ts), name="h(t)", line=dict(color=AZUL, width=3))
    f1.add_scatter(x=tx, y=h0 + m * (tx - t0), name="Tangente: h′(t₀)", line=dict(color=VERDE, width=2))
    f1.add_scatter(x=[t0, t1], y=[h0, h1], name="Secante (Δt)", mode="lines+markers",
                   line=dict(color=AMBAR, width=2, dash="dash"))
    f1.add_scatter(x=[t0], y=[h0], mode="markers", marker=dict(size=10, color=ROJO), showlegend=False)
    f1.update_xaxes(range=[0, T])
    f1.update_yaxes(range=[0, max(H, h1) * 1.08])

    f2 = fig_base("Tasa de ascenso h′(t)", "Tiempo (min)", "ft/min")
    f2.add_scatter(x=ts, y=dh(ts), name="h′(t) = ROC", line=dict(color=VERDE, width=3))
    f2.add_scatter(x=[t0, t1], y=[ms, ms], name="Pendiente media (secante)",
                   line=dict(color=AMBAR, width=2, dash="dash"))
    f2.add_scatter(x=[t0], y=[m], mode="markers", marker=dict(size=10, color=ROJO), name="ROC en t₀")
    f2.update_xaxes(range=[0, T])
    f2.update_yaxes(rangemode="tozero")

    k = st.columns(6)
    k[0].metric("Altitud h(t₀)", f"{fmt(h0, 0)} ft")
    k[1].metric("ROC exacta h′(t₀)", f"{fmt(m, 0)} ft/min")
    k[2].metric("ROC en m/s", f"{fmt(m * 0.00508)} m/s")
    k[3].metric("Pendiente secante", f"{fmt(ms, 0)} ft/min")
    k[4].metric("Error secante", f"{fmt(abs(ms - m) / m * 100)} %")
    k[5].metric("95 % del techo", f"{fmt(3 * tau, 0)} min")
    a, b = st.columns(2)
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    st.info("**Reto:** ¿en qué instante el ROC cae a 500 ft/min? Resuelve h′(t) = 500 y verifícalo con t₀.")


# ------------------------------------------------------------------ Sim 2
def sim_optimizacion():
    st.subheader("2 · Máxima eficiencia aerodinámica (L/D)")
    st.markdown(
        "Con **C_D = C_D0 + k·C_L²** y **k = 1/(π·e·AR)**, la eficiencia es "
        "**L/D = C_L/(C_D0 + k·C_L²)**. El máximo ocurre donde la derivada se anula."
    )
    c = st.columns(5)
    cd0 = c[0].slider("C_D0", 0.015, 0.050, 0.032, 0.001, format="%.3f")
    AR = c[1].slider("Alargamiento AR", 5.0, 20.0, 10.2, 0.1)
    e = c[2].slider("Eficiencia de Oswald e", 0.60, 0.95, 0.75, 0.01)
    cl = c[3].slider("C_L de exploración", 0.05, 1.60, 0.40, 0.01)
    alt = c[4].slider("Altura para planeo (m)", 500, 10000, 3000, 100)

    k = 1 / (np.pi * e * AR)
    ld = lambda x: x / (cd0 + k * x**2)
    dld = lambda x: (cd0 - k * x**2) / (cd0 + k * x**2) ** 2
    clo = np.sqrt(cd0 / k)
    ldm = 1 / (2 * np.sqrt(cd0 * k))
    xs = np.linspace(0.02, 1.6, 300)
    seg = np.array([max(0.02, cl - 0.3), min(1.6, cl + 0.3)])

    f1 = fig_base("Eficiencia aerodinámica L/D", "C_L", "L/D")
    f1.add_scatter(x=xs, y=ld(xs), name="L/D", line=dict(color=AZUL, width=3))
    f1.add_scatter(x=seg, y=ld(cl) + dld(cl) * (seg - cl), name="Tangente en C_L", line=dict(color=VERDE, width=2))
    f1.add_scatter(x=[cl], y=[ld(cl)], mode="markers", marker=dict(size=10, color=ROJO), name="C_L elegido")
    f1.add_scatter(x=[clo], y=[ldm], mode="markers", marker=dict(size=13, color=AMBAR, symbol="star"), name="Máximo")
    f1.update_yaxes(rangemode="tozero")

    f2 = fig_base("Derivada: se anula en el máximo", "C_L", "d(L/D)/dC_L")
    f2.add_scatter(x=xs, y=dld(xs), name="d(L/D)/dC_L", line=dict(color=VERDE, width=3))
    f2.add_hline(y=0, line=dict(color="gray", dash="dot"))
    f2.add_scatter(x=[cl], y=[dld(cl)], mode="markers", marker=dict(size=10, color=ROJO), name="C_L elegido")
    f2.add_scatter(x=[clo], y=[0], mode="markers", marker=dict(size=13, color=AMBAR, symbol="star"), name="Raíz: C_L*")

    m = st.columns(6)
    m[0].metric("k", f"{k:.4f}")
    m[1].metric("C_L* óptimo", f"{clo:.3f}")
    m[2].metric("(L/D) máximo", fmt(ldm))
    m[3].metric("L/D en C_L elegido", fmt(ld(cl)))
    m[4].metric("d(L/D)/dC_L ahí", f"{dld(cl):.3f}")
    m[5].metric(f"Planeo desde {alt} m", f"{fmt(alt * ldm / 1000, 1)} km")
    a, b = st.columns(2)
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    st.markdown(
        "**Derivación:** d(L/D)/dC_L = (C_D0 − k·C_L²)/(C_D0 + k·C_L²)² = 0 ⇒ "
        "C_L* = √(C_D0/k) y (L/D)máx = 1/(2√(C_D0·k)). En el óptimo, C_D = 2·C_D0 "
        "(resistencia parásita = inducida)."
    )
    st.info("**Reto:** ¿cómo cambia (L/D)máx si duplicas AR? ¿Por qué los planeadores tienen alas tan largas?")


# ------------------------------------------------------------------ Sim 3
def sim_combustible():
    st.subheader("3 · Combustible de una misión como integral del flujo")
    st.markdown(
        "El flujo es **ṁ(t) = F_cr + (F_cl − F_cr)·e^(−t/τ)**. El combustible total es **∫₀ᵀ ṁ(t) dt**. "
        "Compara métodos numéricos con la integral exacta."
    )
    c = st.columns(6)
    Fcl = c[0].slider("F_cl ascenso (kg/min)", 0.4, 2.0, 0.85, 0.05)
    Fcr = c[1].slider("F_cr crucero (kg/min)", 0.3, 1.2, 0.55, 0.01)
    tau = c[2].slider("τ (min)", 3, 30, 10)
    T = c[3].slider("Duración T (min)", 30, 360, 120, 5)
    n = c[4].slider("Subintervalos n (par)", 2, 100, 8, 2)
    metodo = c[5].selectbox("Método", ["Riemann izquierda", "Riemann derecha", "Punto medio", "Trapecio", "Simpson 1/3"], index=3)

    f = lambda t: Fcr + (Fcl - Fcr) * np.exp(-t / tau)
    exacto = Fcr * T + (Fcl - Fcr) * tau * (1 - np.exp(-T / tau))
    num = integrar(metodo, f, 0, T, n)
    err = abs(num - exacto)
    h = T / n
    ts = np.linspace(0, T, 300)

    f1 = fig_base("Flujo ṁ(t) y aproximación", "Tiempo (min)", "kg/min")
    f1.add_scatter(x=ts, y=f(ts), name="ṁ(t)", line=dict(color=AZUL, width=3))
    relleno, borde = "rgba(251,191,36,0.28)", dict(color=AMBAR, width=1)
    if metodo in ("Riemann izquierda", "Riemann derecha", "Punto medio"):
        for i in range(n):
            x0 = i * h
            xs_ = {"Riemann izquierda": x0, "Riemann derecha": x0 + h, "Punto medio": x0 + h / 2}[metodo]
            f1.add_shape(type="rect", x0=x0, x1=x0 + h, y0=0, y1=f(xs_), fillcolor=relleno, line=borde)
    elif metodo == "Trapecio":
        for i in range(n):
            x0, x1 = i * h, (i + 1) * h
            f1.add_shape(type="path", path=f"M {x0},0 L {x0},{f(x0)} L {x1},{f(x1)} L {x1},0 Z",
                         fillcolor=relleno, line=borde)
    else:
        px, py = [], []
        for q in range(n // 2):
            x0, x1, x2 = 2 * q * h, 2 * q * h + h, 2 * q * h + 2 * h
            xx = np.linspace(x0, x2, 13)
            l0 = (xx - x1) * (xx - x2) / ((x0 - x1) * (x0 - x2))
            l1 = (xx - x0) * (xx - x2) / ((x1 - x0) * (x1 - x2))
            l2 = (xx - x0) * (xx - x1) / ((x2 - x0) * (x2 - x1))
            px += list(xx)
            py += list(f(x0) * l0 + f(x1) * l1 + f(x2) * l2)
        f1.add_scatter(x=px, y=py, name="Parábolas de Simpson", fill="tozeroy", fillcolor=relleno,
                       line=dict(color=AMBAR, width=1.5))
    f1.update_yaxes(rangemode="tozero")

    ns = list(range(2, 101, 2))
    es = [max(abs(integrar(metodo, f, 0, T, q) - exacto), 1e-12) for q in ns]
    f2 = fig_base("Convergencia del método (escala log)", "n subintervalos", "Error (kg)")
    f2.add_scatter(x=ns, y=es, name="Error absoluto", line=dict(color=ROJO, width=3))
    f2.add_scatter(x=[n], y=[max(err, 1e-12)], mode="markers", marker=dict(size=11, color=AMBAR), name="n actual")
    f2.update_xaxes(type="log")
    f2.update_yaxes(type="log")

    m = st.columns(5)
    m[0].metric("Numérica", f"{fmt(num, 2)} kg")
    m[1].metric("Integral exacta", f"{fmt(exacto, 2)} kg")
    m[2].metric("Error absoluto", f"{fmt(err, 4)} kg")
    m[3].metric("Error relativo", f"{fmt(err / exacto * 100, 4)} %")
    m[4].metric("Volumen Avgas 100LL", f"{fmt(exacto / 0.72, 1)} L")
    a, b = st.columns(2)
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    st.markdown(
        "**Integral exacta:** F_cr·T + (F_cl − F_cr)·τ·(1 − e^(−T/τ)). "
        "Error ∝ 1/n (Riemann), 1/n² (punto medio y trapecio), 1/n⁴ (Simpson)."
    )
    st.info("**Reto:** ¿cuántos subintervalos necesita cada método para un error relativo menor a 0,01 %?")


# ------------------------------------------------------------------ Sim 4
def forma(tipo, u, lam):
    if tipo == "Elíptica":
        return np.sqrt(np.clip(1 - u**2, 0, None))
    if tipo == "Rectangular":
        return np.ones_like(u)
    return 1 - (1 - lam) * u


def ala(tipo, lam, b, masa, nf, N=400):
    half = b / 2
    Ltot = nf * masa * G
    u = np.linspace(0, 1, 4001)
    s = forma(tipo, u, lam)
    I0 = _trapz(s, u)
    I1 = _trapz(u * s, u)
    L0 = (Ltot / 2) / (half * I0)
    y = np.linspace(0, half, N + 1)
    Lp = L0 * forma(tipo, 2 * y / b, lam)
    h = half / N
    V = np.zeros(N + 1)
    M = np.zeros(N + 1)
    for i in range(N - 1, -1, -1):
        V[i] = V[i + 1] + (Lp[i] + Lp[i + 1]) / 2 * h
        M[i] = M[i + 1] + (V[i] + V[i + 1]) / 2 * h
    return dict(Ltot=Ltot, L0=L0, y=y, Lp=Lp, V=V, M=M, half=half, N=N,
                Mroot=L0 * half**2 * I1, ycp=half * I1 / I0)


def sim_ala():
    st.subheader("4 · Sustentación en el ala: cortante y momento flector")
    st.markdown(
        "La sustentación por unidad de envergadura **L′(y)** suma n·W. Cortante: **V(y) = ∫ L′ ds** y "
        "momento: **M(y) = ∫ L′(s)(s − y) ds**, de y hasta la punta (sin peso del ala)."
    )
    c = st.columns(6)
    b = c[0].slider("Envergadura b (m)", 8.0, 40.0, 8.8, 0.1)
    masa = c[1].slider("Masa (kg)", 500, 50000, 998, 2)
    nf = c[2].slider("Factor de carga n (g)", 1.0, 6.0, 1.0, 0.1)
    tipo = c[3].selectbox("Distribución", ["Elíptica", "Rectangular", "Lineal (taper)"])
    lam = c[4].slider("Taper (punta/raíz)", 0.0, 1.0, 0.4, 0.05, disabled=(tipo != "Lineal (taper)"))
    pc = c[5].slider("Estación y₀ (% semiala)", 0, 100, 0)

    w = ala(tipo, lam, b, masa, nf)
    i0 = min(w["N"], round(pc / 100 * w["N"]))
    y0 = w["y"][i0]
    y, Lp = w["y"], w["Lp"] / 1000

    f1 = fig_base("Sustentación por unidad de envergadura", "Posición y (m)", "kN/m")
    f1.add_scatter(x=np.concatenate([-y[::-1], y[1:]]), y=np.concatenate([Lp[::-1], Lp[1:]]), name="L′(y)",
                   fill="tozeroy", fillcolor="rgba(56,189,248,0.22)", line=dict(color=AZUL, width=3))
    f1.add_scatter(x=y[i0:], y=Lp[i0:], name="Región y₀ → punta", fill="tozeroy",
                   fillcolor="rgba(251,191,36,0.35)", line=dict(color=AMBAR, width=2))
    f1.update_yaxes(rangemode="tozero")

    f2 = fig_base("Cortante y momento a lo largo del semiala", "Distancia desde la raíz (m)", "Cortante (kN)",
                  yaxis2=dict(title="Momento (kN·m)", overlaying="y", side="right", rangemode="tozero", showgrid=False))
    f2.add_scatter(x=y, y=w["V"] / 1000, name="Cortante V(y) [kN]", line=dict(color=VERDE, width=3))
    f2.add_scatter(x=y, y=w["M"] / 1000, name="Momento M(y) [kN·m]", yaxis="y2", line=dict(color=ROJO, width=3))
    f2.add_scatter(x=[y0], y=[w["V"][i0] / 1000], mode="markers", marker=dict(size=10, color=VERDE), showlegend=False)
    f2.add_scatter(x=[y0], y=[w["M"][i0] / 1000], mode="markers", yaxis="y2", marker=dict(size=10, color=ROJO), showlegend=False)
    f2.update_layout(yaxis=dict(rangemode="tozero"))

    m = st.columns(6)
    m[0].metric("n·W", f"{fmt(w['Ltot'] / 1000)} kN")
    m[1].metric("L₀", f"{fmt(w['L0'] / 1000)} kN/m")
    m[2].metric("M en la raíz", f"{fmt(w['Mroot'] / 1000)} kN·m")
    m[3].metric("Centro de presión", f"{fmt(w['ycp'])} m ({fmt(w['ycp'] / w['half'] * 100, 1)} %)")
    m[4].metric("V(y₀)", f"{fmt(w['V'][i0] / 1000)} kN")
    m[5].metric("M(y₀)", f"{fmt(w['M'][i0] / 1000)} kN·m")
    if tipo == "Elíptica":
        Man = w["Ltot"] * b / (3 * np.pi)
        st.caption(f"Control analítico n·W·b/(3π) = {fmt(Man / 1000)} kN·m · "
                   f"diferencia con la integral numérica: {fmt(abs(w['M'][0] - Man) / Man * 100, 3)} %")
    a, bcol = st.columns(2)
    a.plotly_chart(f1, width="stretch")
    bcol.plotly_chart(f2, width="stretch")

    filas = []
    for t in ["Elíptica", "Rectangular", "Lineal (taper)"]:
        q = ala(t, lam, b, masa, nf)
        filas.append({"Distribución": t if t != "Lineal (taper)" else f"Lineal (taper = {lam:.2f})",
                      "M raíz (kN·m)": round(q["Mroot"] / 1000, 2),
                      "C.P. (% semiala)": round(q["ycp"] / q["half"] * 100, 1)})
    st.markdown("**Momento en la raíz con la misma sustentación total**")
    st.dataframe(filas, hide_index=True, width="stretch")
    st.info("**Reto:** demuestra que ∫₀¹ √(1 − u²) du = π/4 y ∫₀¹ u·√(1 − u²) du = 1/3, y compáralo con lo numérico.")


# ------------------------------------------------------------------ principal
st.title("✈️ Cálculo en el Aire")
st.caption("Cuatro simulaciones de Cálculo Diferencial e Integral aplicadas a la aviación")
st.info(
    "**Aeronave de referencia: T-90 Calima** (entrenador básico de la FAC). Datos publicados por "
    "[Airforce Technology](https://www.airforce-technology.com/projects/calima-t-90-trainer-aircraft/): "
    "envergadura 8,8 m · área alar 7,6 m² (AR ≈ 10,2) · peso máx. operativo 998 kg · motor Lycoming IO-390 de 210 hp · "
    "altitud máx. 18.000 ft · crucero 296 km/h.\n\n"
    "**Supuestos didácticos** (no publicados): C_D0, e, la constante de ascenso τ y los flujos de combustible "
    "(gasolina de aviación 100LL). Ajusta los controles para otra aeronave."
)

SIMS = {
    "1 · Derivada: tasa de ascenso": sim_derivada,
    "2 · Optimización: máx. L/D": sim_optimizacion,
    "3 · Integral: combustible": sim_combustible,
    "4 · Integral: carga en el ala": sim_ala,
}
pestanas = st.tabs(list(SIMS))
for pestana, simulacion in zip(pestanas, SIMS.values()):
    with pestana:
        simulacion()

st.markdown("---")
st.markdown(f"<div style='text-align:center;opacity:.75'>{CREDITO}</div>", unsafe_allow_html=True)
