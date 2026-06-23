"""
modelagem.py — Etapa 4 do KDD: Mineração (Modelagem) e Avaliação
Projeto: Predição de Patologias Cardíacas (UCMF/RHP)

Depende de: data/processed/UCMF_features.csv (saída da Etapa 2 — features.py)
Saída:
  - reports/modelagem_summary.txt        (log textual completo)
  - reports/metricas_modelos.csv         (tabela de métricas, 2 cenários)
  - reports/figures/04_*.png             (comparação, matrizes de confusão,
                                          importância de atributos)

Objetivo: prever NORMAL_X_ANORMAL (Normal × Anormal) a partir dos dados
clínicos/antropométricos. Conforme exigência do enunciado, todas as métricas
são reportadas em DUAS versões: COM a variável SOPRO e SEM a variável SOPRO.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/UCMF_features.csv")
FIGURES_DIR   = Path("reports/figures")
SUMMARY_PATH  = Path("reports/modelagem_summary.txt")
METRICS_CSV   = Path("reports/metricas_modelos.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.25
N_FOLDS      = 5
TARGET       = "NORMAL_X_ANORMAL"
# Classe positiva = "Anormal" (1): é a classe clinicamente relevante de
# detectar (presença de patologia). Precisão/recall/F1 são reportados
# para essa classe positiva.

# Atributos candidatos (excluídos: ID, datas, flags de auditoria, texto
# livre de alta cardinalidade — HDA1/HDA2/MOTIVO1/MOTIVO2/CONVENIO — e a
# própria variável-alvo).
NUMERICAS = [
    "IDADE", "PESO", "ALTURA", "IMC", "FC",
    "PA_SISTOLICA", "PA_DIASTOLICA",
    "PERCENTIL_ALTURA", "PERCENTIL_IMC",
]
CATEGORICAS_BASE = [
    "SEXO", "FAIXA_ETARIA",
    "PULSOS", "B2",
    "PPA_CALCULADO", "IMC_CLASSIFICACAO",
]
# SOPRO é tratada à parte para permitir os dois cenários (com/sem SOPRO).
SOPRO_COL = "SOPRO"


# ─────────────────────────────────────────────
# Utilitários de log e figuras
# ─────────────────────────────────────────────
def log(linhas: list, texto=""):
    print(texto)
    linhas.append(str(texto))


def salvar_fig(fig, nome: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{nome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[salvar_fig] {path}")


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PREPARO DA AMOSTRA SUPERVISIONADA
# ─────────────────────────────────────────────
def carregar_dados(log_lines: list):
    """
    Carrega o dataset de features e mantém apenas registros rotulados
    (NORMAL_X_ANORMAL ∈ {Normal, Anormal}). Mapeia o alvo para 0/1
    (Anormal = 1 = classe positiva).
    """
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    log(log_lines, f"[dados] Dataset de features carregado: {df.shape}")

    rotulados = df[df[TARGET].isin(["Normal", "Anormal"])].copy()
    n_sem_rotulo = len(df) - len(rotulados)
    log(log_lines, f"[dados] Registros sem rótulo descartados (alvo NaN): {n_sem_rotulo}")
    log(log_lines, f"[dados] Amostra supervisionada: {len(rotulados)} registros")

    y = (rotulados[TARGET] == "Anormal").astype(int)
    log(log_lines, f"[dados] Distribuição do alvo  →  "
                   f"Normal(0): {(y == 0).sum()}  |  Anormal(1): {(y == 1).sum()}  "
                   f"(prevalência Anormal = {y.mean()*100:.1f}%)")
    return rotulados, y


# ─────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO (imputação + escala + one-hot)
# ─────────────────────────────────────────────
def construir_preprocessador(num_cols, cat_cols) -> ColumnTransformer:
    """
    Numéricas: imputação pela mediana + padronização (z-score).
    Categóricas: imputação por categoria constante "Não informado" +
                 codificação one-hot (ignora categorias não vistas no teste).

    Tudo encapsulado em ColumnTransformer para que as estatísticas de
    imputação/escala sejam aprendidas SOMENTE no conjunto de treino de cada
    fold/partição — evitando vazamento de informação (data leakage).
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Não informado")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ─────────────────────────────────────────────
# 3. MODELOS
# ─────────────────────────────────────────────
def construir_modelos() -> dict:
    """Cinco classificadores clássicos de mineração de dados."""
    return {
        "Árvore de Decisão": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=8, min_samples_leaf=20),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Regressão Logística": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE),
    }


# ─────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────
def metricas_teste(y_true, y_pred, y_proba) -> dict:
    """Métricas no conjunto de teste (classe positiva = Anormal = 1)."""
    return {
        "Acurácia":         accuracy_score(y_true, y_pred),
        "Precisão_Anormal": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_Anormal":   recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_Anormal":       f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC":          roc_auc_score(y_true, y_proba),
    }


# ─────────────────────────────────────────────
# 5. AVALIAÇÃO DE UM CENÁRIO (com ou sem SOPRO)
# ─────────────────────────────────────────────
def avaliar_cenario(nome_cenario, X_train, X_test, y_train, y_test,
                    num_cols, cat_cols, log_lines):
    """
    Roda os 5 modelos no cenário definido por (num_cols, cat_cols):
      - validação cruzada estratificada (5 folds) no TREINO → estabilidade
      - ajuste final no treino e avaliação no TESTE (held-out)
    Retorna (DataFrame de métricas, nome do melhor modelo, pipelines, feature_cols).
    """
    feature_cols = num_cols + cat_cols
    Xtr = X_train[feature_cols]
    Xte = X_test[feature_cols]

    pre = construir_preprocessador(num_cols, cat_cols)
    modelos = construir_modelos()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    log(log_lines, "\n" + "=" * 64)
    log(log_lines, f"CENÁRIO: {nome_cenario}")
    log(log_lines, f"  Nº de atributos de entrada: {len(feature_cols)} "
                   f"({len(num_cols)} numéricos + {len(cat_cols)} categóricos)")
    log(log_lines, "=" * 64)

    linhas, pipelines = [], {}
    for nome_m, est in modelos.items():
        pipe = Pipeline([("prep", pre), ("clf", est)])

        cvres = cross_validate(pipe, Xtr, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(Xtr, y_train)
        y_pred  = pipe.predict(Xte)
        y_proba = pipe.predict_proba(Xte)[:, 1]

        m = metricas_teste(y_test, y_pred, y_proba)
        m["CV_F1_média"] = cvres["test_f1"].mean()
        m["CV_F1_dp"]    = cvres["test_f1"].std()
        m["Modelo"]      = nome_m
        linhas.append(m)
        pipelines[nome_m] = pipe

    res = pd.DataFrame(linhas).set_index("Modelo")
    res = res[["Acurácia", "Precisão_Anormal", "Recall_Anormal",
               "F1_Anormal", "F1_macro", "ROC_AUC", "CV_F1_média", "CV_F1_dp"]]

    log(log_lines, "\n── Métricas no conjunto de TESTE (classe positiva = Anormal) ──")
    log(log_lines, res.round(4).to_string())

    # Melhor modelo: maior F1 (Anormal) na validação cruzada (seleção que
    # não enxerga o conjunto de teste → evita superajuste de seleção).
    melhor = res["CV_F1_média"].idxmax()
    log(log_lines, f"\n>> Melhor modelo do cenário (por CV-F1): {melhor}  "
                   f"(CV-F1 = {res.loc[melhor,'CV_F1_média']:.4f} ± "
                   f"{res.loc[melhor,'CV_F1_dp']:.4f}; "
                   f"F1-teste = {res.loc[melhor,'F1_Anormal']:.4f})")
    return res, melhor, pipelines, feature_cols


# ─────────────────────────────────────────────
# 6. ARTEFATOS DE INTERPRETAÇÃO DO MELHOR MODELO
# ─────────────────────────────────────────────
def matriz_confusao(pipe, X_test, y_test, feature_cols, titulo, nome_arquivo, log_lines):
    y_pred = pipe.predict(X_test[feature_cols])
    cm = confusion_matrix(y_test, y_pred)
    log(log_lines, f"\n── Matriz de confusão — {titulo} ──")
    log(log_lines, pd.DataFrame(
        cm, index=["Real:Normal", "Real:Anormal"],
        columns=["Prev:Normal", "Prev:Anormal"]).to_string())
    log(log_lines, "\n" + classification_report(
        y_test, y_pred, target_names=["Normal", "Anormal"], digits=3))

    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Anormal"],
                yticklabels=["Normal", "Anormal"], ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    salvar_fig(fig, nome_arquivo)


def importancia_atributos(pipe, X_test, y_test, feature_cols, titulo,
                          nome_arquivo, log_lines):
    """
    Importância por permutação (model-agnóstica) medida no conjunto de teste,
    agregada por atributo original (não por dummy one-hot). Mede a queda de
    F1 ao embaralhar cada atributo.
    """
    r = permutation_importance(
        pipe, X_test[feature_cols], y_test,
        scoring="f1", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = (pd.Series(r.importances_mean, index=feature_cols)
           .sort_values(ascending=False))

    log(log_lines, f"\n── Importância por permutação (queda de F1) — {titulo} ──")
    log(log_lines, imp.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    imp_plot = imp.head(12).iloc[::-1]
    ax.barh(imp_plot.index, imp_plot.values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Queda média de F1 ao permutar o atributo")
    ax.set_title(titulo)
    plt.tight_layout()
    salvar_fig(fig, nome_arquivo)
    return imp


def grafico_comparacao(res_com, res_sem, log_lines):
    """Barras lado a lado: F1(Anormal) de cada modelo, com vs sem SOPRO."""
    modelos = list(res_com.index)
    x = np.arange(len(modelos))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w/2, res_com["F1_Anormal"], w, label="Com SOPRO",
                color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w/2, res_sem["F1_Anormal"], w, label="Sem SOPRO",
                color="#C44E52", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=20, ha="right")
    ax.set_ylabel("F1-score (classe Anormal)")
    ax.set_title("Desempenho por modelo — COM vs SEM a variável SOPRO")
    ax.set_ylim(0, 1.0)
    ax.legend()
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    salvar_fig(fig, "04_comparacao_modelos")


# ─────────────────────────────────────────────
# 7. PIPELINE COMPLETO
# ─────────────────────────────────────────────
def pipeline_modelagem():
    log_lines = []
    log(log_lines, "=" * 64)
    log(log_lines, "PIPELINE DE MODELAGEM E AVALIAÇÃO — UCMF/RHP (Etapa 4 do KDD)")
    log(log_lines, "=" * 64)

    # --- 1. Dados + split estratificado único (idêntico nos dois cenários) ---
    X, y = carregar_dados(log_lines)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log(log_lines, f"[split] Treino: {len(X_train)}  |  Teste: {len(X_test)}  "
                   f"(estratificado, test_size={TEST_SIZE}, seed={RANDOM_STATE})")

    cat_com = CATEGORICAS_BASE + [SOPRO_COL]
    cat_sem = CATEGORICAS_BASE

    # --- 2. Cenário COM SOPRO ---
    res_com, melhor_com, pipes_com, cols_com = avaliar_cenario(
        "COM a variável SOPRO", X_train, X_test, y_train, y_test,
        NUMERICAS, cat_com, log_lines)

    # --- 3. Cenário SEM SOPRO ---
    res_sem, melhor_sem, pipes_sem, cols_sem = avaliar_cenario(
        "SEM a variável SOPRO", X_train, X_test, y_train, y_test,
        NUMERICAS, cat_sem, log_lines)

    # --- 4. Artefatos do melhor modelo em cada cenário ---
    log(log_lines, "\n" + "=" * 64)
    log(log_lines, "INTERPRETAÇÃO DO MELHOR MODELO")
    log(log_lines, "=" * 64)

    matriz_confusao(pipes_com[melhor_com], X_test, y_test, cols_com,
                    f"{melhor_com} — COM SOPRO",
                    "04_matriz_confusao_com_sopro", log_lines)
    importancia_atributos(pipes_com[melhor_com], X_test, y_test, cols_com,
                          f"{melhor_com} — COM SOPRO",
                          "04_importancia_com_sopro", log_lines)

    matriz_confusao(pipes_sem[melhor_sem], X_test, y_test, cols_sem,
                    f"{melhor_sem} — SEM SOPRO",
                    "04_matriz_confusao_sem_sopro", log_lines)
    importancia_atributos(pipes_sem[melhor_sem], X_test, y_test, cols_sem,
                          f"{melhor_sem} — SEM SOPRO",
                          "04_importancia_sem_sopro", log_lines)

    grafico_comparacao(res_com, res_sem, log_lines)

    # --- 5. Tabela consolidada (com vs sem SOPRO) ---
    log(log_lines, "\n" + "=" * 64)
    log(log_lines, "RESUMO CONSOLIDADO — COM vs SEM SOPRO (conjunto de teste)")
    log(log_lines, "=" * 64)
    consolidado = pd.concat(
        {"COM_SOPRO": res_com, "SEM_SOPRO": res_sem}, axis=1)
    log(log_lines, consolidado.round(4).to_string())

    delta = (res_com["F1_Anormal"] - res_sem["F1_Anormal"])
    log(log_lines, "\n── Queda de F1 (Anormal) ao remover SOPRO, por modelo ──")
    log(log_lines, delta.round(4).sort_values(ascending=False).to_string())

    # --- 6. Persistência ---
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    consolidado.round(4).to_csv(METRICS_CSV, encoding="utf-8")
    print(f"\n[pipeline] Métricas salvas em: {METRICS_CSV}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"[pipeline] Log salvo em: {SUMMARY_PATH}")

    log(log_lines, "\n" + "=" * 64)
    log(log_lines, "MODELAGEM CONCLUÍDA.")
    log(log_lines, "=" * 64)
    return res_com, res_sem, melhor_com, melhor_sem


if __name__ == "__main__":
    pipeline_modelagem()
