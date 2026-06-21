"""
eda.py — Etapa 3 do KDD: Análise Exploratória (EDA) e Visualização
Projeto: Predição de Patologias Cardíacas (UCMF/RHP)

Depende de: data/processed/UCMF_features.csv (saída da Etapa 2 — features.py)
Saída: reports/figures/*.png + reports/eda_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/UCMF_features.csv")
FIGURES_DIR   = Path("reports/figures")
SUMMARY_PATH  = Path("reports/eda_summary.txt")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
PALETA = {"Normal": "#4C72B0", "Anormal": "#C44E52"}


# ─────────────────────────────────────────────
# 0. UTILITÁRIOS
# ─────────────────────────────────────────────

def salvar_fig(fig, nome: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{nome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[salvar_fig] {path}")


def log(linhas: list, texto: str):
    print(texto)
    linhas.append(texto)


# ─────────────────────────────────────────────
# 1. ESTATÍSTICAS DESCRITIVAS
# ─────────────────────────────────────────────

def estatisticas_descritivas(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "1. ESTATÍSTICAS DESCRITIVAS")
    log(log_lines, "=" * 60)

    numericas = ["PESO", "ALTURA", "IMC", "IDADE", "PA_SISTOLICA",
                 "PA_DIASTOLICA", "FC", "PERCENTIL_ALTURA", "PERCENTIL_IMC"]
    numericas = [c for c in numericas if c in df.columns]

    log(log_lines, "\n── Variáveis numéricas ──")
    log(log_lines, df[numericas].describe().round(2).to_string())

    categoricas = ["SEXO", "PULSOS", "B2", "SOPRO", "NORMAL_X_ANORMAL",
                   "IMC_CLASSIFICACAO", "PPA_CALCULADO", "FAIXA_ETARIA"]
    categoricas = [c for c in categoricas if c in df.columns]

    log(log_lines, "\n── Variáveis categóricas (top valores) ──")
    for c in categoricas:
        log(log_lines, f"\n{c}:")
        log(log_lines, df[c].value_counts(dropna=False).to_string())


# ─────────────────────────────────────────────
# 2. DISTRIBUIÇÃO DA VARIÁVEL-ALVO
# ─────────────────────────────────────────────

def distribuicao_alvo(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "2. DISTRIBUIÇÃO DA VARIÁVEL-ALVO (NORMAL_X_ANORMAL)")
    log(log_lines, "=" * 60)

    vc = df["NORMAL_X_ANORMAL"].value_counts(dropna=False)
    pct = (vc / len(df) * 100).round(2)
    log(log_lines, pd.DataFrame({"contagem": vc, "pct_%": pct}).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["NORMAL_X_ANORMAL"].value_counts()
    cores = [PALETA.get(c, "#999999") for c in counts.index]
    counts.plot.bar(ax=ax, color=cores, edgecolor="white")
    ax.set_title("Distribuição: NORMAL × ANORMAL")
    ax.set_ylabel("Contagem")
    ax.set_xlabel("")
    for i, v in enumerate(counts):
        ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
    plt.xticks(rotation=0)
    salvar_fig(fig, "03_distribuicao_alvo")


# ─────────────────────────────────────────────
# 3. CORRELAÇÕES ENTRE NUMÉRICAS
# ─────────────────────────────────────────────

def correlacoes(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "3. CORRELAÇÕES ENTRE VARIÁVEIS NUMÉRICAS")
    log(log_lines, "=" * 60)

    numericas = ["PESO", "ALTURA", "IMC", "IDADE", "PA_SISTOLICA",
                 "PA_DIASTOLICA", "FC"]
    numericas = [c for c in numericas if c in df.columns]

    corr = df[numericas].corr(method="pearson")
    log(log_lines, corr.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Matriz de Correlação — Variáveis Numéricas")
    salvar_fig(fig, "03_matriz_correlacao")


# ─────────────────────────────────────────────
# 4. HISTOGRAMAS
# ─────────────────────────────────────────────

def histogramas(df: pd.DataFrame):
    numericas = ["PESO", "ALTURA", "IMC", "IDADE", "PA_SISTOLICA",
                 "PA_DIASTOLICA", "FC"]
    numericas = [c for c in numericas if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        df[col].dropna().hist(bins=40, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col)
    for j in range(len(numericas), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Histogramas — Variáveis Numéricas", y=1.02)
    plt.tight_layout()
    salvar_fig(fig, "03_histogramas")


# ─────────────────────────────────────────────
# 5. BOXPLOTS POR SEXO
# ─────────────────────────────────────────────

def boxplots_por_sexo(df: pd.DataFrame):
    numericas = ["PESO", "ALTURA", "IMC", "PA_SISTOLICA", "PA_DIASTOLICA", "FC"]
    numericas = [c for c in numericas if c in df.columns]

    df_plot = df[df["SEXO"].isin(["M", "F"])]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        sns.boxplot(data=df_plot, x="SEXO", y=col, ax=axes[i],
                     palette={"M": "#4C72B0", "F": "#DD8452"})
        axes[i].set_title(f"{col} por SEXO")
    plt.suptitle("Boxplots por Sexo", y=1.02)
    plt.tight_layout()
    salvar_fig(fig, "03_boxplots_sexo")


# ─────────────────────────────────────────────
# 6. DISPERSÃO IMC × IDADE e PA × IDADE
# ─────────────────────────────────────────────

def dispersao_idade(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    df_plot = df[df["NORMAL_X_ANORMAL"].isin(["Normal", "Anormal"])]

    sns.scatterplot(data=df_plot, x="IDADE", y="IMC", hue="NORMAL_X_ANORMAL",
                     palette=PALETA, alpha=0.4, s=15, ax=axes[0])
    axes[0].set_title("IMC × IDADE")

    sns.scatterplot(data=df_plot, x="IDADE", y="PA_SISTOLICA", hue="NORMAL_X_ANORMAL",
                     palette=PALETA, alpha=0.4, s=15, ax=axes[1])
    axes[1].set_title("PA Sistólica × IDADE")

    sns.scatterplot(data=df_plot, x="IDADE", y="PA_DIASTOLICA", hue="NORMAL_X_ANORMAL",
                     palette=PALETA, alpha=0.4, s=15, ax=axes[2])
    axes[2].set_title("PA Diastólica × IDADE")

    plt.suptitle("Dispersão: IMC e PA em função da IDADE", y=1.02)
    plt.tight_layout()
    salvar_fig(fig, "03_dispersao_idade")


# ─────────────────────────────────────────────
# 7. B2 / SOPRO × DIAGNÓSTICO
# ─────────────────────────────────────────────

def categoricas_vs_diagnostico(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "4. COMPARAÇÃO NORMAL × ANORMAL (PULSOS, B2, SOPRO)")
    log(log_lines, "=" * 60)

    df_plot = df[df["NORMAL_X_ANORMAL"].isin(["Normal", "Anormal"])]
    cats = ["PULSOS", "B2", "SOPRO"]
    cats = [c for c in cats if c in df.columns]

    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 5))
    if len(cats) == 1:
        axes = [axes]

    for ax, col in zip(axes, cats):
        tab = pd.crosstab(df_plot[col], df_plot["NORMAL_X_ANORMAL"], normalize="index") * 100
        log(log_lines, f"\n{col} × NORMAL_X_ANORMAL (% por linha):")
        log(log_lines, tab.round(1).to_string())

        tab.plot.bar(stacked=True, ax=ax, color=[PALETA["Anormal"], PALETA["Normal"]],
                     edgecolor="white")
        ax.set_title(f"{col} × Diagnóstico (%)")
        ax.set_ylabel("% de pacientes")
        ax.legend(title="Diagnóstico", loc="upper right")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Sinais Clínicos × Diagnóstico (NORMAL × ANORMAL)", y=1.02)
    plt.tight_layout()
    salvar_fig(fig, "03_categoricas_vs_diagnostico")


# ─────────────────────────────────────────────
# 8. HDA1 / HDA2 — MOTIVOS MAIS COMUNS POR DIAGNÓSTICO
# ─────────────────────────────────────────────

def hda_vs_diagnostico(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "5. HDA1 — TOP MOTIVOS POR DIAGNÓSTICO")
    log(log_lines, "=" * 60)

    if "HDA1" not in df.columns:
        return

    df_plot = df[df["NORMAL_X_ANORMAL"].isin(["Normal", "Anormal"]) & df["HDA1"].notna()]

    top_geral = df_plot["HDA1"].value_counts().head(8).index
    df_top = df_plot[df_plot["HDA1"].isin(top_geral)]

    tab = pd.crosstab(df_top["HDA1"], df_top["NORMAL_X_ANORMAL"])
    log(log_lines, tab.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    tab.plot.barh(ax=ax, color=[PALETA["Anormal"], PALETA["Normal"]], edgecolor="white")
    ax.set_title("HDA1 (top 8) × Diagnóstico")
    ax.set_xlabel("Contagem")
    plt.tight_layout()
    salvar_fig(fig, "03_hda1_vs_diagnostico")


# ─────────────────────────────────────────────
# 9. PPA E IMC_CLASSIFICACAO × DIAGNÓSTICO
# ─────────────────────────────────────────────

def features_derivadas_vs_diagnostico(df: pd.DataFrame, log_lines: list):
    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "6. PPA_CALCULADO e IMC_CLASSIFICACAO × DIAGNÓSTICO")
    log(log_lines, "=" * 60)

    df_plot = df[df["NORMAL_X_ANORMAL"].isin(["Normal", "Anormal"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col in zip(axes, ["PPA_CALCULADO", "IMC_CLASSIFICACAO"]):
        sub = df_plot[df_plot[col] != "Não Calculado"]
        tab = pd.crosstab(sub[col], sub["NORMAL_X_ANORMAL"], normalize="index") * 100
        log(log_lines, f"\n{col} × NORMAL_X_ANORMAL (% por linha, excluindo 'Não Calculado'):")
        log(log_lines, tab.round(1).to_string())

        tab.plot.bar(stacked=True, ax=ax, color=[PALETA["Anormal"], PALETA["Normal"]],
                     edgecolor="white")
        ax.set_title(f"{col} × Diagnóstico (%)")
        ax.set_ylabel("% de pacientes")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Features Derivadas × Diagnóstico", y=1.02)
    plt.tight_layout()
    salvar_fig(fig, "03_features_derivadas_vs_diagnostico")


# ─────────────────────────────────────────────
# 10. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def pipeline_eda(input_path: Path = FEATURES_PATH) -> None:
    log_lines = []

    log(log_lines, "=" * 60)
    log(log_lines, "PIPELINE DE EDA — UCMF/RHP")
    log(log_lines, "=" * 60)

    df = pd.read_csv(input_path, low_memory=False)
    log(log_lines, f"[pipeline] Dataset carregado: {df.shape}")

    estatisticas_descritivas(df, log_lines)
    distribuicao_alvo(df, log_lines)
    correlacoes(df, log_lines)
    histogramas(df)
    boxplots_por_sexo(df)
    dispersao_idade(df)
    categoricas_vs_diagnostico(df, log_lines)
    hda_vs_diagnostico(df, log_lines)
    features_derivadas_vs_diagnostico(df, log_lines)

    log(log_lines, "\n" + "=" * 60)
    log(log_lines, "EDA CONCLUÍDA — gráficos salvos em reports/figures/")
    log(log_lines, "=" * 60)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[pipeline] Resumo textual salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    pipeline_eda()
