"""Checkpoint 3 — Dashboard de Churn de Clientes (Telco).

Executar:  streamlit run app.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dados import (CAMINHO_METRICAS, CAMINHO_MODELO, CATEGORICAS,
                       FAIXAS_PERMANENCIA, FONTE_NOME, FONTE_URL, NUMERICAS,
                       ROTULOS, carregar_dados)

st.set_page_config(page_title="Dashboard de Churn — Telco", page_icon="📉",
                   layout="wide")

COR_FICOU = "#4C78A8"
COR_SAIU = "#E45756"
CORES_CHURN = {"Não": COR_FICOU, "Sim": COR_SAIU}


# ---------------------------------------------------------------- dados/modelo
@st.cache_data
def dados() -> pd.DataFrame:
    return carregar_dados()


@st.cache_resource
def modelo_e_metricas():
    try:
        pipe = joblib.load(CAMINHO_MODELO)
        with open(CAMINHO_METRICAS, encoding="utf-8") as f:
            met = json.load(f)
    except Exception:
        # Se o .joblib não abrir (ex.: versão diferente do scikit-learn), retreina
        from src.treinar_modelo import treinar
        pipe = treinar()
        with open(CAMINHO_METRICAS, encoding="utf-8") as f:
            met = json.load(f)
    return pipe, met


def grafico(fig):
    """Exibe um gráfico Plotly com vírgula decimal (padrão brasileiro)."""
    fig.update_layout(separators=",.")
    st.plotly_chart(fig, width="stretch")


def taxa_churn(d: pd.DataFrame) -> float:
    return (d["churn"] == "Sim").mean() if len(d) else 0.0


def usd(v: float) -> str:
    return f"US$ {num(v)}"


def pct(v: float) -> str:
    return f"{v:.1%}".replace(".", ",")


def num(v: float, casas: int = 0) -> str:
    """Número no padrão brasileiro: 1.234,5"""
    txt = f"{v:,.{casas}f}"
    return txt.replace(",", "X").replace(".", ",").replace("X", ".")


df = dados()
pipe, met = modelo_e_metricas()

# -------------------------------------------------------------------- filtros
st.sidebar.header("🔎 Filtros")
st.sidebar.caption("Valem para as abas Visão geral e Exploração.")


def multiselect(col):
    opcoes = sorted(df[col].unique()) if col != "faixa_permanencia" else FAIXAS_PERMANENCIA
    return st.sidebar.multiselect(ROTULOS[col], opcoes, default=opcoes)


f_contrato = multiselect("tipo_contrato")
f_internet = multiselect("tipo_internet")
f_faixa = multiselect("faixa_permanencia")
f_pagamento = multiselect("forma_pagamento")
f_genero = multiselect("genero")
f_idoso = multiselect("idoso")
vmin, vmax = float(df["valor_mensal"].min()), float(df["valor_mensal"].max())
f_valor = st.sidebar.slider(ROTULOS["valor_mensal"], vmin, vmax, (vmin, vmax))

dff = df[
    df["tipo_contrato"].isin(f_contrato)
    & df["tipo_internet"].isin(f_internet)
    & df["faixa_permanencia"].isin(f_faixa)
    & df["forma_pagamento"].isin(f_pagamento)
    & df["genero"].isin(f_genero)
    & df["idoso"].isin(f_idoso)
    & df["valor_mensal"].between(*f_valor)
]
st.sidebar.markdown(f"**{num(len(dff))}** de {num(len(df))} clientes selecionados")

# -------------------------------------------------------------------- cabeçalho
st.title("📉 Dashboard de Churn de Clientes — Telecom")
st.markdown(
    "Análise exploratória e preditiva do cancelamento de clientes (churn) de "
    "uma operadora de telecomunicações. **Fonte dos dados:** "
    f"[{FONTE_NOME}]({FONTE_URL}) — 7.043 clientes, 21 variáveis."
)

aba_visao, aba_exp, aba_modelo, aba_sim, aba_dados = st.tabs(
    ["📊 Visão geral", "🔬 Exploração", "🤖 Modelo preditivo",
     "🧮 Simulador", "🗂️ Fonte dos dados"])

if dff.empty:
    for aba in (aba_visao, aba_exp):
        aba.warning("Nenhum cliente com os filtros selecionados.")

# ---------------------------------------------------------------- visão geral
with aba_visao:
    if not dff.empty:
        saiu = dff[dff["churn"] == "Sim"]
        receita = dff["valor_mensal"].sum()
        receita_perdida = saiu["valor_mensal"].sum()
        base = taxa_churn(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Clientes", num(len(dff)))
        c2.metric("Taxa de churn", pct(taxa_churn(dff)),
                  delta=f"{(taxa_churn(dff) - base) * 100:+.1f}".replace(".", ",")
                  + " p.p. vs. base",
                  delta_color="inverse")
        c3.metric("Receita mensal", usd(receita))
        c4.metric("Receita mensal perdida", usd(receita_perdida),
                  help="Soma do valor mensal dos clientes que cancelaram")
        c5.metric("Permanência média", f"{num(dff['meses_contrato'].mean(), 1)} meses")

        c6, c7, c8 = st.columns(3)
        c6.metric("Ticket médio (quem ficou)",
                  usd(dff.loc[dff["churn"] == "Não", "valor_mensal"].mean() or 0))
        c7.metric("Ticket médio (quem saiu)",
                  usd(saiu["valor_mensal"].mean() if len(saiu) else 0))
        c8.metric("% da receita perdida",
                  pct(receita_perdida / receita if receita else 0))

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Churn por tipo de contrato")
            t = (dff.groupby("tipo_contrato")["churn"]
                 .apply(lambda s: (s == "Sim").mean()).reset_index(name="taxa"))
            fig = px.bar(t, x="tipo_contrato", y="taxa", text_auto=".1%",
                         labels={"tipo_contrato": "", "taxa": "Taxa de churn"},
                         category_orders={"tipo_contrato": ["Mensal", "Anual", "Bienal"]},
                         color_discrete_sequence=[COR_SAIU])
            fig.update_yaxes(tickformat=".0%")
            grafico(fig)
        with g2:
            st.subheader("Clientes que ficaram × saíram")
            cont = dff["churn"].value_counts().reset_index()
            cont.columns = ["churn", "clientes"]
            cont["churn"] = cont["churn"].map({"Não": "Ficou", "Sim": "Saiu"})
            fig = px.pie(cont, names="churn", values="clientes", hole=0.55,
                         color="churn",
                         color_discrete_map={"Ficou": COR_FICOU, "Saiu": COR_SAIU})
            grafico(fig)

        st.subheader("Taxa de churn por faixa de permanência")
        t = (dff.groupby("faixa_permanencia")["churn"]
             .apply(lambda s: (s == "Sim").mean()).reindex(FAIXAS_PERMANENCIA)
             .reset_index(name="taxa").dropna())
        fig = px.line(t, x="faixa_permanencia", y="taxa", markers=True,
                      labels={"faixa_permanencia": "", "taxa": "Taxa de churn"},
                      color_discrete_sequence=[COR_SAIU])
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        grafico(fig)
        st.info("**Leitura:** o churn se concentra nos clientes com contrato mensal "
                "e nos primeiros 12 meses de relacionamento. Contratos anuais e "
                "bienais quase não cancelam.")

# ------------------------------------------------------------------ exploração
with aba_exp:
    if not dff.empty:
        st.subheader("Gráficos interativos")
        e1, e2 = st.columns(2)
        with e1:
            var = st.selectbox(
                "Taxa de churn por…",
                [c for c in CATEGORICAS if c != "tipo_contrato"],
                format_func=ROTULOS.get, index=6)  # abre em "Tipo de internet"
            t = (dff.groupby(var)["churn"].agg(
                taxa=lambda s: (s == "Sim").mean(), clientes="size").reset_index())
            fig = px.bar(t, x=var, y="taxa", text_auto=".1%",
                         hover_data={"clientes": True},
                         labels={var: ROTULOS[var], "taxa": "Taxa de churn"},
                         color_discrete_sequence=[COR_SAIU])
            fig.update_yaxes(tickformat=".0%")
            grafico(fig)
        with e2:
            var_num = st.selectbox("Distribuição de…", NUMERICAS,
                                   format_func=ROTULOS.get)
            fig = px.histogram(dff, x=var_num, color="churn", barmode="overlay",
                               nbins=40, opacity=0.65,
                               color_discrete_map=CORES_CHURN,
                               labels={var_num: ROTULOS[var_num], "churn": "Cancelou"})
            grafico(fig)

        st.subheader("Valor mensal × tempo como cliente")
        amostra = dff.sample(min(len(dff), 2500), random_state=1)
        fig = px.scatter(amostra, x="meses_contrato", y="valor_mensal",
                         color="churn", opacity=0.6,
                         color_discrete_map=CORES_CHURN,
                         hover_data=["tipo_contrato", "tipo_internet"],
                         labels={"meses_contrato": ROTULOS["meses_contrato"],
                                 "valor_mensal": ROTULOS["valor_mensal"],
                                 "churn": "Cancelou"})
        grafico(fig)

        st.subheader("Gráficos estáticos")
        s1, s2 = st.columns(2)
        with s1:
            fig, ax = plt.subplots(figsize=(6, 4))
            dados_box = [dff.loc[dff["churn"] == k, "valor_mensal"] for k in ("Não", "Sim")]
            bp = ax.boxplot(dados_box, patch_artist=True)
            ax.set_xticks([1, 2], ["Ficou", "Saiu"])
            for patch, cor in zip(bp["boxes"], [COR_FICOU, COR_SAIU]):
                patch.set_facecolor(cor)
                patch.set_alpha(0.7)
            ax.set_title("Valor mensal por status")
            ax.set_ylabel("US$")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        with s2:
            tab = pd.crosstab(dff["tipo_internet"], dff["tipo_contrato"],
                              values=(dff["churn"] == "Sim"), aggfunc="mean")
            tab = tab.reindex(columns=[c for c in ["Mensal", "Anual", "Bienal"]
                                       if c in tab.columns])
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(tab.values, cmap="Reds", vmin=0, vmax=0.6)
            ax.set_xticks(range(len(tab.columns)), tab.columns)
            ax.set_yticks(range(len(tab.index)), tab.index)
            for i in range(tab.shape[0]):
                for j in range(tab.shape[1]):
                    v = tab.values[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                                color="white" if v > 0.35 else "black")
            ax.set_title("Taxa de churn: internet × contrato")
            fig.colorbar(im, ax=ax, format=lambda x, _: f"{x:.0%}")
            st.pyplot(fig)

        st.subheader("Correlação entre variáveis numéricas")
        corr_df = dff[NUMERICAS].assign(churn=(dff["churn"] == "Sim").astype(int))
        corr = corr_df.corr().rename(index=ROTULOS, columns=ROTULOS)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        grafico(fig)

# ------------------------------------------------------------- modelo preditivo
with aba_modelo:
    res = met["resultados"]
    escolhido = met["modelo_escolhido"]
    st.markdown(
        f"Foram treinados **{len(res)} modelos** de classificação para prever se um "
        f"cliente vai cancelar. Dados divididos em {num(met['n_treino'])} clientes de "
        f"treino e {num(met['n_teste'])} de teste (80/20, estratificado). "
        f"**Modelo escolhido: {escolhido}** — critério: {met['criterio']}."
    )
    r = res[escolhido]
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, nome, chave in [(m1, "Acurácia", "acuracia"), (m2, "Precisão", "precisao"),
                             (m3, "Recall", "recall"), (m4, "F1", "f1"),
                             (m5, "ROC-AUC", "roc_auc")]:
        col.metric(nome, f"{r[chave]:.3f}".replace(".", ","))

    st.subheader("Comparação dos modelos")
    tabela = pd.DataFrame({
        n: {"Acurácia": v["acuracia"], "Precisão": v["precisao"],
            "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"]}
        for n, v in res.items()}).T
    st.dataframe(tabela.style.format(precision=3, decimal=",").highlight_max(axis=0, color="#d4edda"),
                 width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Curva ROC")
        fig = go.Figure()
        for n, v in res.items():
            fig.add_trace(go.Scatter(x=v["roc"]["fpr"], y=v["roc"]["tpr"], mode="lines",
                                     name=f"{n} (AUC {v['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(xaxis_title="Taxa de falsos positivos",
                          yaxis_title="Taxa de verdadeiros positivos",
                          legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98))
        grafico(fig)
    with c2:
        st.subheader(f"Matriz de confusão — {escolhido}")
        mc = r["matriz_confusao"]
        fig = px.imshow(mc, text_auto=True, color_continuous_scale="Blues",
                        x=["Previsto: fica", "Previsto: sai"],
                        y=["Real: ficou", "Real: saiu"])
        fig.update_layout(coloraxis_showscale=False)
        grafico(fig)
        tn, fp, fn, tp = mc[0][0], mc[0][1], mc[1][0], mc[1][1]
        st.caption(f"O modelo identificou {tp} dos {tp + fn} clientes que realmente "
                   f"cancelaram no teste, com {fp} alarmes falsos.")

    st.subheader("Variáveis mais importantes")
    imp = pd.DataFrame(met["importancias"], columns=["variavel", "importancia"])
    imp["variavel"] = imp["variavel"].map(ROTULOS)
    fig = px.bar(imp.head(10).iloc[::-1], x="importancia", y="variavel",
                 orientation="h", text_auto=".1%",
                 labels={"importancia": "Importância relativa", "variavel": ""},
                 color_discrete_sequence=[COR_FICOU])
    fig.update_xaxes(tickformat=".0%")
    grafico(fig)

# ------------------------------------------------------------------- simulador
with aba_sim:
    st.markdown("Preencha o perfil de um cliente para estimar a **probabilidade de "
                "cancelamento** com o modelo treinado.")

    def op(col):
        return sorted(df[col].unique())

    with st.form("simulador"):
        a, b, c = st.columns(3)
        entrada = {}
        with a:
            st.markdown("**Contrato e cobrança**")
            entrada["tipo_contrato"] = st.selectbox(ROTULOS["tipo_contrato"],
                                                    ["Mensal", "Anual", "Bienal"])
            entrada["meses_contrato"] = st.slider(ROTULOS["meses_contrato"], 0, 72, 6)
            entrada["valor_mensal"] = st.slider(ROTULOS["valor_mensal"], 18.0, 120.0, 80.0)
            entrada["forma_pagamento"] = st.selectbox(ROTULOS["forma_pagamento"],
                                                      op("forma_pagamento"))
            entrada["fatura_digital"] = st.radio(ROTULOS["fatura_digital"],
                                                 ["Sim", "Não"], horizontal=True)
        with b:
            st.markdown("**Serviços**")
            entrada["tipo_internet"] = st.selectbox(ROTULOS["tipo_internet"],
                                                    op("tipo_internet"))
            sem_net = entrada["tipo_internet"] == "Não"
            for col in ["seguranca_online", "backup_online", "protecao_aparelho",
                        "suporte_tecnico", "streaming_tv", "streaming_filmes"]:
                entrada[col] = ("Sem internet" if sem_net else
                                st.radio(ROTULOS[col], ["Não", "Sim"], horizontal=True))
            if sem_net:
                st.caption("Sem internet: serviços online desativados.")
        with c:
            st.markdown("**Perfil**")
            entrada["genero"] = st.radio(ROTULOS["genero"], op("genero"), horizontal=True)
            entrada["idoso"] = st.radio(ROTULOS["idoso"], ["Não", "Sim"], horizontal=True)
            entrada["tem_parceiro"] = st.radio(ROTULOS["tem_parceiro"], ["Não", "Sim"],
                                               horizontal=True)
            entrada["tem_dependentes"] = st.radio(ROTULOS["tem_dependentes"],
                                                  ["Não", "Sim"], horizontal=True)
            entrada["servico_telefone"] = st.radio(ROTULOS["servico_telefone"],
                                                   ["Sim", "Não"], horizontal=True)
            entrada["multiplas_linhas"] = (
                "Sem telefone" if entrada["servico_telefone"] == "Não" else
                st.radio(ROTULOS["multiplas_linhas"], ["Não", "Sim"], horizontal=True))
        enviar = st.form_submit_button("Calcular risco", type="primary")

    if enviar:
        entrada["valor_total"] = entrada["valor_mensal"] * entrada["meses_contrato"]
        X_novo = pd.DataFrame([entrada])[NUMERICAS + CATEGORICAS]
        p = float(pipe.predict_proba(X_novo)[0, 1])
        nivel, cor = (("Alto", COR_SAIU) if p >= 0.6 else
                      ("Médio", "#F2A541") if p >= 0.35 else ("Baixo", "#54A24B"))
        r1, r2 = st.columns([1, 2])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": cor},
                       "steps": [{"range": [0, 35], "color": "#eef6ec"},
                                 {"range": [35, 60], "color": "#fdf3e3"},
                                 {"range": [60, 100], "color": "#fbe9e9"}]}))
            fig.update_layout(height=260, margin=dict(t=30, b=0, l=40, r=40))
            grafico(fig)
        with r2:
            st.markdown(f"### Risco **{nivel}** de cancelamento")
            st.write(f"Probabilidade estimada: **{pct(p)}** "
                     f"(média da base: {pct(taxa_churn(df))}).")
            dicas = []
            if entrada["tipo_contrato"] == "Mensal":
                dicas.append("Oferecer migração para contrato anual ou bienal.")
            if entrada["meses_contrato"] <= 12:
                dicas.append("Cliente novo: incluir em programa de boas-vindas.")
            if entrada["tipo_internet"] == "Fibra óptica" and entrada["suporte_tecnico"] == "Não":
                dicas.append("Oferecer suporte técnico ou segurança online como bônus.")
            if entrada["forma_pagamento"] == "Boleto eletrônico":
                dicas.append("Incentivar débito automático ou cartão.")
            if dicas and p >= 0.35:
                st.markdown("**Ações sugeridas de retenção:**\n" +
                            "\n".join(f"- {d}" for d in dicas))

# -------------------------------------------------------------- fonte dos dados
with aba_dados:
    st.subheader("Fonte dos dados")
    st.markdown(f"""
- **Dataset:** [{FONTE_NOME}]({FONTE_URL})
- **Origem:** conjunto de exemplo publicado pela IBM, amplamente usado em estudos de churn.
- **Tamanho:** {num(len(df))} clientes × 21 variáveis originais.
- **Alvo:** `Churn` — se o cliente cancelou o serviço no último mês.
- **Tratamento:** colunas e categorias traduzidas para português; `TotalCharges`
  convertido para número (11 clientes com 0 meses tinham o campo vazio → 0);
  `SeniorCitizen` convertido de 0/1 para Não/Sim; criada a coluna *faixa de permanência*.
""")
    st.subheader("Dicionário de variáveis")
    st.dataframe(pd.DataFrame({"Coluna": list(ROTULOS),
                               "Descrição": list(ROTULOS.values())}),
                 hide_index=True, width="stretch")
    st.subheader("Amostra dos dados filtrados")
    st.dataframe(dff.head(200), width="stretch")
    st.download_button("⬇️ Baixar dados filtrados (CSV)",
                       dff.to_csv(index=False).encode("utf-8"),
                       "churn_filtrado.csv", "text/csv")
