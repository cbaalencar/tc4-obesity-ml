import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .block-container { padding-top: 2rem; }
    .result-card {
        background: #1A1D27;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #1A1D27;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover { background: #2563EB; }
    h1, h2, h3 { color: #E5E7EB; }
    .sidebar .sidebar-content { background: #1A1D27; }
</style>
""", unsafe_allow_html=True)

# ── Paleta de cores por classe ────────────────────────────────────────────────
PALETTE = {
    'Insufficient_Weight': ('#3B82F6', '🔵', 'Peso Insuficiente'),
    'Normal_Weight':       ('#22C55E', '🟢', 'Peso Normal'),
    'Overweight_Level_I':  ('#F59E0B', '🟡', 'Sobrepeso Nível I'),
    'Overweight_Level_II': ('#F97316', '🟠', 'Sobrepeso Nível II'),
    'Obesity_Type_I':      ('#EF4444', '🔴', 'Obesidade Tipo I'),
    'Obesity_Type_II':     ('#B91C1C', '🔴', 'Obesidade Tipo II'),
    'Obesity_Type_III':    ('#7F1D1D', '🔴', 'Obesidade Tipo III'),
}

TARGET_ORDER = list(PALETTE.keys())

# ── Carregamento dos arquivos ─────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    with open('modelo_final.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_metadata.json') as f:
        meta = json.load(f)
    return modelo, scaler, meta

try:
    modelo, scaler, meta = carregar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    st.error(f"Erro ao carregar modelo: {e}")
    st.info("Certifique-se de que modelo_final.pkl, scaler.pkl e feature_metadata.json estão na mesma pasta do app.py")

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🏥 Sistema de Predição de Obesidade")
st.markdown(
    "Ferramenta de apoio à decisão clínica para classificação do nível de obesidade "
    "com base em dados demográficos e hábitos do paciente."
)
st.divider()

if not modelo_ok:
    st.stop()

# ── Sidebar: Formulário do paciente ──────────────────────────────────────────
with st.sidebar:
    st.header("📋 Dados do Paciente")
    st.markdown("Preencha os campos abaixo:")

    st.subheader("Dados Pessoais")
    genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
    idade  = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
    altura = st.number_input("Altura (metros)", min_value=1.40, max_value=2.20,
                             value=1.70, step=0.01, format="%.2f")
    peso   = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0,
                             value=70.0, step=0.5, format="%.1f")

    st.subheader("Histórico e Hábitos Alimentares")
    historico_familiar = st.selectbox("Histórico familiar de excesso de peso?", ["Não", "Sim"])
    favc = st.selectbox("Consome alimentos calóricos com frequência?", ["Não", "Sim"])
    fcvc = st.selectbox("Frequência de vegetais nas refeições",
                        [1, 2, 3],
                        format_func=lambda x: {1: "Raramente", 2: "Às vezes", 3: "Sempre"}[x])
    ncp  = st.selectbox("Refeições principais por dia",
                        [1, 2, 3, 4],
                        index=2,
                        format_func=lambda x: f"{x} refeição{'ões' if x > 1 else ''}")
    caec = st.selectbox("Come entre as refeições?",
                        ["no", "Sometimes", "Frequently", "Always"],
                        index=1,
                        format_func=lambda x: {
                            "no": "Não", "Sometimes": "Às vezes",
                            "Frequently": "Frequentemente", "Always": "Sempre"
                        }[x])

    st.subheader("Estilo de Vida")
    smoke  = st.selectbox("Fumante?", ["Não", "Sim"])
    ch2o   = st.selectbox("Litros de água por dia",
                          [1, 2, 3],
                          index=1,
                          format_func=lambda x: {1: "Menos de 1L", 2: "Entre 1L e 2L", 3: "Mais de 2L"}[x])
    scc    = st.selectbox("Monitora calorias ingeridas?", ["Não", "Sim"])
    faf    = st.selectbox("Dias de atividade física por semana",
                          [0, 1, 2, 3],
                          format_func=lambda x: {0: "Nenhum", 1: "1–2 dias", 2: "2–4 dias", 3: "4–5 dias"}[x])
    tue    = st.selectbox("Horas diárias com dispositivos tecnológicos",
                          [0, 1, 2],
                          format_func=lambda x: {0: "0–2h", 1: "3–5h", 2: "Mais de 5h"}[x])
    calc   = st.selectbox("Consome álcool?",
                          ["no", "Sometimes", "Frequently", "Always"],
                          index=1,
                          format_func=lambda x: {
                              "no": "Não", "Sometimes": "Às vezes",
                              "Frequently": "Frequentemente", "Always": "Sempre"
                          }[x])
    mtrans = st.selectbox("Meio de transporte principal",
                          ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
                          index=3,
                          format_func=lambda x: {
                              "Automobile": "Automóvel", "Bike": "Bicicleta",
                              "Motorbike": "Moto", "Public_Transportation": "Transporte Público",
                              "Walking": "Caminhada"
                          }[x])

    prever = st.button("🔍 Analisar Paciente")

# ── Função de pré-processamento ───────────────────────────────────────────────
def preparar_entrada(meta):
    bmi = peso / (altura ** 2)

    # Monta dicionário base
    entrada = {
        'Gender':          1 if genero == "Masculino" else 0,
        'Age':             float(idade),
        'Height':          float(altura),
        'Weight':          float(peso),
        'family_history':  1 if historico_familiar == "Sim" else 0,
        'FAVC':            1 if favc == "Sim" else 0,
        'FCVC':            int(fcvc),
        'NCP':             int(ncp),
        'CAEC':            ["no", "Sometimes", "Frequently", "Always"].index(caec),
        'SMOKE':           1 if smoke == "Sim" else 0,
        'CH2O':            int(ch2o),
        'SCC':             1 if scc == "Sim" else 0,
        'FAF':             int(faf),
        'TUE':             int(tue),
        'CALC':            ["no", "Sometimes", "Frequently", "Always"].index(calc),
        'BMI':             round(bmi, 2),
    }

    # One-hot MTRANS
    for cat in meta['mtrans_categories']:
        entrada[f'MTRANS_{cat}'] = 1 if mtrans == cat else 0

    # Monta DataFrame na ordem correta das features
    df_entrada = pd.DataFrame([entrada])[meta['feature_names']]

    # Aplica scaler nas numéricas
    df_entrada[meta['numeric_scaled']] = scaler.transform(
        df_entrada[meta['numeric_scaled']]
    )

    return df_entrada, bmi

# ── Conteúdo principal ────────────────────────────────────────────────────────
col_info, col_result = st.columns([1, 1])

with col_info:
    st.subheader("ℹ️ Sobre o Modelo")
    acc = meta.get('acuracia_teste', 0.9811)
    f1  = meta.get('f1_macro_teste', 0.9806)
    alg = meta.get('modelo_nome', 'Random Forest')

    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritmo", alg.replace(" ", "\n"))
    c2.metric("Acurácia", f"{acc*100:.1f}%")
    c3.metric("F1-Score", f"{f1*100:.1f}%")

    st.markdown("---")
    st.subheader("📊 Classes de Risco")
    for cls, (cor, icone, label) in PALETTE.items():
        st.markdown(
            f'<span style="color:{cor}; font-size:1.1rem">{icone} {label}</span>',
            unsafe_allow_html=True
        )

with col_result:
    st.subheader("📈 Resultado da Análise")

    if not prever:
        st.info("Preencha os dados do paciente na barra lateral e clique em **Analisar Paciente**.")
    else:
        with st.spinner("Processando..."):
            df_entrada, bmi = preparar_entrada(meta)

            pred_idx   = modelo.predict(df_entrada)[0]
            pred_proba = modelo.predict_proba(df_entrada)[0]
            pred_class = TARGET_ORDER[pred_idx]

            cor, icone, label = PALETTE[pred_class]
            confianca = pred_proba[pred_idx] * 100

        # Resultado principal
        st.markdown(
            f"""<div class="result-card">
                <h1 style="color:{cor}; font-size:2.5rem; margin:0">{icone}</h1>
                <h2 style="color:{cor}; margin:0.5rem 0">{label}</h2>
                <p style="color:#9CA3AF; font-size:1rem">
                    Confiança: <strong style="color:{cor}">{confianca:.1f}%</strong>
                    &nbsp;|&nbsp; IMC: <strong style="color:#E5E7EB">{bmi:.1f} kg/m²</strong>
                </p>
            </div>""",
            unsafe_allow_html=True
        )

        # Probabilidades por classe
        st.markdown("**Probabilidade por classe:**")
        fig, ax = plt.subplots(figsize=(7, 3.5),
                               facecolor='#1A1D27')
        ax.set_facecolor('#1A1D27')
        cores = [PALETTE[c][0] for c in TARGET_ORDER]
        labels_pt = [PALETTE[c][2] for c in TARGET_ORDER]
        alphas = [1.0 if i == pred_idx else 0.35 for i in range(len(TARGET_ORDER))]

        bars = ax.barh(labels_pt[::-1],
                       [pred_proba[i] * 100 for i in range(len(TARGET_ORDER))][::-1],
                       color=[cores[i] for i in range(len(TARGET_ORDER))][::-1],
                       alpha=1.0, edgecolor='none', height=0.55)

        for bar, alpha in zip(bars, [alphas[i] for i in range(len(TARGET_ORDER))][::-1]):
            bar.set_alpha(alpha)
            val = bar.get_width()
            if val > 1:
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=9, color='#E5E7EB')

        ax.set_xlim(0, 110)
        ax.set_xlabel('Probabilidade (%)', color='#9CA3AF', fontsize=9)
        ax.tick_params(colors='#9CA3AF', labelsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(axis='x', alpha=0.2, color='#4B5563')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Seção de Explicabilidade ──────────────────────────────────────────────────
if prever and modelo_ok:
    st.divider()
    st.subheader("🔎 Fatores Mais Relevantes para a Predição")

    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=meta['feature_names'])
    else:
        imp = pd.Series(
            np.abs(modelo.coef_).mean(axis=0),
            index=meta['feature_names']
        )

    imp = imp.sort_values(ascending=False).head(10)

    nomes_pt = {
        'BMI': 'IMC', 'Weight': 'Peso', 'Height': 'Altura', 'Age': 'Idade',
        'Gender': 'Gênero', 'family_history': 'Histórico Familiar',
        'FCVC': 'Vegetais nas refeições', 'NCP': 'Refeições por dia',
        'CAEC': 'Come entre refeições', 'SMOKE': 'Fumante',
        'CH2O': 'Consumo de água', 'SCC': 'Monitora calorias',
        'FAF': 'Atividade física', 'TUE': 'Uso de tecnologia',
        'CALC': 'Consumo de álcool', 'FAVC': 'Alimentos calóricos',
        'MTRANS_Automobile': 'Transporte: Automóvel',
        'MTRANS_Bike': 'Transporte: Bicicleta',
        'MTRANS_Motorbike': 'Transporte: Moto',
        'MTRANS_Public_Transportation': 'Transporte Público',
        'MTRANS_Walking': 'Transporte: Caminhada',
    }

    labels = [nomes_pt.get(f, f) for f in imp.index]

    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#1A1D27')
    ax2.set_facecolor('#1A1D27')
    cores_imp = ['#3B82F6' if i < 3 else '#475569' for i in range(len(imp))]
    ax2.barh(labels[::-1], imp.values[::-1],
             color=cores_imp[::-1], edgecolor='none', height=0.55)
    ax2.set_xlabel('Importância relativa', color='#9CA3AF', fontsize=9)
    ax2.tick_params(colors='#9CA3AF', labelsize=9)
    ax2.spines[:].set_visible(False)
    ax2.grid(axis='x', alpha=0.2, color='#4B5563')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.caption(
        "Os três fatores destacados em azul são os de maior peso na classificação. "
        "IMC e Peso são determinantes primários; os demais representam fatores comportamentais e genéticos."
    )

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Sistema desenvolvido como trabalho acadêmico — FIAP Pós-Tech Data Analytics | Tech Challenge Fase 4. "
    "Não substitui avaliação médica profissional."
)
