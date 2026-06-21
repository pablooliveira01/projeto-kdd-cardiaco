"""
limpeza.py — Etapa 1 do KDD: Seleção e Limpeza dos Dados
Projeto: Predição de Patologias Cardíacas (UCMF/RHP)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Caminhos padrão
# ─────────────────────────────────────────────
RAW_PATH       = Path("data/raw/UCMF_raw.csv")
PROCESSED_PATH = Path("data/processed/UCMF_clean.csv")
LOG_PATH       = Path("data/processed/limpeza_log.txt")


# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────

def carregar_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Carrega o CSV raw e padroniza os nomes das colunas
    (remove espaços, substitui por underscore, uppercase).
    """
    df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)

    # Padronizar nomes: strip + underscore + upper
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.upper()
    )

    # Renomear colunas com nome composto problemático
    renomear = {
        "PA_SISTOLICA":      "PA_SISTOLICA",   # já ok após upper
        "PA_DIASTOLICA":     "PA_DIASTOLICA",
        "NORMAL_X_ANORMAL":  "NORMAL_X_ANORMAL",
        "HDA_1":             "HDA1",
        "HDA_2":             "HDA2",           # já vem como HDA2 no raw
    }
    df.rename(columns=renomear, inplace=True)

    print(f"[carregar] {len(df)} linhas, {len(df.columns)} colunas.")
    return df


# ─────────────────────────────────────────────
# 2. DATAS
# ─────────────────────────────────────────────

def converter_datas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte Atendimento e DN para datetime (dayfirst=True).
    Calcula IDADE_CALC em anos a partir da diferença de datas e
    sinaliza inconsistências com a IDADE informada.
    """
    for col in ["ATENDIMENTO", "DN"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Idade recalculada em anos (fracionada)
    df["IDADE_CALC"] = (
        (df["ATENDIMENTO"] - df["DN"]).dt.days / 365.25
    ).round(4)

    # Flag de inconsistência: diferença > 1 ano ou IDADE negativa
    df["FLAG_IDADE_INCONSISTENTE"] = (
        df["IDADE"].isna() |
        df["IDADE_CALC"].isna() |
        (df["IDADE"] < 0) |
        (df["IDADE_CALC"] < 0) |
        ((df["IDADE_CALC"] - df["IDADE"]).abs() > 1.0)
    )

    n = df["FLAG_IDADE_INCONSISTENTE"].sum()
    print(f"[datas] {n} registros com IDADE inconsistente ou negativa.")
    return df


# ─────────────────────────────────────────────
# 3. VARIÁVEIS NUMÉRICAS
# ─────────────────────────────────────────────

def limpar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Peso e Altura iguais a 0 → NaN
    - ALTURA < 40cm (implausível) → NaN
    - IDADE < 0 → NaN  (usar IDADE_CALC quando disponível)
    - IDADE > 19 → flag (fora do escopo pediátrico, mantido no dataset)
    - PA_SISTOLICA / PA_DIASTOLICA <= 0 → NaN
    - PA_SISTOLICA < PA_DIASTOLICA (inversão fisiológica) → NaN nos dois
    - IMC: recalcular a partir de Peso/Altura² e comparar com informado
    - IMC_CALC > 40 (implausível) → PESO/ALTURA/IMC → NaN
    - Flags de outlier para revisão manual
    """
    # --- Peso / Altura ---
    df["PESO"]   = pd.to_numeric(df["PESO"],   errors="coerce")
    df["ALTURA"] = pd.to_numeric(df["ALTURA"], errors="coerce")

    df.loc[df["PESO"]   <= 0, "PESO"]   = np.nan
    df.loc[df["ALTURA"] <= 0, "ALTURA"] = np.nan

    # Altura implausivelmente baixa (< 40cm): nem um recém-nascido prematuro
    # mede menos que isso. Provável erro de digitação (dígito faltando,
    # ex: "117" registrado como "17"). Gera IMC astronômico se não tratado.
    df["FLAG_ALTURA_IMPLAUSIVEL"] = df["ALTURA"].notna() & (df["ALTURA"] < 40)
    n_alt = df["FLAG_ALTURA_IMPLAUSIVEL"].sum()
    df.loc[df["FLAG_ALTURA_IMPLAUSIVEL"], "ALTURA"] = np.nan
    print(f"[numericas] {n_alt} registros com ALTURA < 40cm (implausível) → NaN.")

    # --- IDADE ---
    df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
    df.loc[df["IDADE"] < 0, "IDADE"] = np.nan

    # Preencher IDADE com IDADE_CALC onde IDADE é NaN
    mask_sem_idade = df["IDADE"].isna() & df["IDADE_CALC"].notna() & (df["IDADE_CALC"] >= 0)
    df.loc[mask_sem_idade, "IDADE"] = df.loc[mask_sem_idade, "IDADE_CALC"].round(2)
    print(f"[numericas] IDADE preenchida por IDADE_CALC em {mask_sem_idade.sum()} registros.")

    # Idade fora do escopo do estudo (0–19 anos, conforme RHP/UCMF)
    df["FLAG_IDADE_FORA_ESCOPO"] = df["IDADE"].notna() & (df["IDADE"] > 19)
    print(f"[numericas] {df['FLAG_IDADE_FORA_ESCOPO'].sum()} registros com IDADE > 19 anos (fora do escopo pediátrico).")

    # --- PA ---
    df["PA_SISTOLICA"]  = pd.to_numeric(df["PA_SISTOLICA"],  errors="coerce")
    df["PA_DIASTOLICA"] = pd.to_numeric(df["PA_DIASTOLICA"], errors="coerce")

    # PA negativa ou zero
    df.loc[df["PA_SISTOLICA"]  <= 0, "PA_SISTOLICA"]  = np.nan
    df.loc[df["PA_DIASTOLICA"] <= 0, "PA_DIASTOLICA"] = np.nan

    # Inversão PAS < PAD
    inv = df["PA_SISTOLICA"] < df["PA_DIASTOLICA"]
    df.loc[inv, ["PA_SISTOLICA", "PA_DIASTOLICA"]] = np.nan
    print(f"[numericas] {inv.sum()} registros com PA invertida → NaN.")

    # PA implausível (> 250 mmHg): mesmo a HAS estágio 2 mais grave em
    # pediatria não chega perto disso. Provável erro de digitação
    # (dígito extra, ex: "99" registrado como "990").
    df["FLAG_PA_IMPLAUSIVEL"] = (
        (df["PA_SISTOLICA"].notna() & (df["PA_SISTOLICA"] > 250)) |
        (df["PA_DIASTOLICA"].notna() & (df["PA_DIASTOLICA"] > 250))
    )
    n_pa = df["FLAG_PA_IMPLAUSIVEL"].sum()
    df.loc[df["FLAG_PA_IMPLAUSIVEL"], ["PA_SISTOLICA", "PA_DIASTOLICA"]] = np.nan
    print(f"[numericas] {n_pa} registros com PA > 250 mmHg (implausível) → NaN.")

    # --- IMC ---
    df["IMC"] = pd.to_numeric(df["IMC"], errors="coerce")

    altura_m = df["ALTURA"] / 100  # cm → m
    df["IMC_CALC"] = (df["PESO"] / (altura_m ** 2)).round(2)

    df["FLAG_IMC_DIVERGENTE"] = (
        df["IMC"].notna() &
        df["IMC_CALC"].notna() &
        ((df["IMC_CALC"] - df["IMC"]).abs() > 1.0)
    )
    n_div = df["FLAG_IMC_DIVERGENTE"].sum()
    print(f"[numericas] {n_div} registros com IMC informado divergindo >1 do calculado.")

    # IMC implausível (> 40): segunda camada de segurança contra outliers
    # de digitação em PESO ou ALTURA que escapam do filtro de altura mínima
    # (ex: peso de adulto registrado para criança). IMC > 40 em pediatria
    # é extremamente raro.
    df["FLAG_IMC_IMPLAUSIVEL"] = df["IMC_CALC"].notna() & (df["IMC_CALC"] > 40)
    n_imc = df["FLAG_IMC_IMPLAUSIVEL"].sum()
    df.loc[df["FLAG_IMC_IMPLAUSIVEL"], ["PESO", "ALTURA", "IMC_CALC"]] = np.nan
    print(f"[numericas] {n_imc} registros com IMC_CALC > 40 (implausível) → PESO/ALTURA/IMC → NaN.")

    # Usar IMC_CALC como referência oficial
    df["IMC"] = df["IMC_CALC"]

    # --- FC ---
    df["FC"] = pd.to_numeric(df["FC"], errors="coerce")

    # Outliers fisiológicos (FC fora de 30–300 bpm)
    df["FLAG_FC_OUTLIER"] = df["FC"].notna() & ((df["FC"] < 30) | (df["FC"] > 300))
    df.loc[df["FLAG_FC_OUTLIER"], "FC"] = np.nan
    print(f"[numericas] {df['FLAG_FC_OUTLIER'].sum()} registros com FC fora de 30-300 bpm → NaN.")

    return df


# ─────────────────────────────────────────────
# 4. VARIÁVEIS CATEGÓRICAS
# ─────────────────────────────────────────────

_MAP_SEXO = {
    "m": "M", "masculino": "M", "male": "M",
    "f": "F", "feminino": "F", "female": "F",
    "indeterminado": "I",
}

_MAP_PULSOS = {
    "normais": "Normais",
    "diminuídos": "Diminuídos", "diminuidos": "Diminuídos",
    "ausentes": "Ausentes",
    "amplos": "Amplos",
    "femorais diminuidos": "Femorais Diminuídos",
    "femorais diminuídos": "Femorais Diminuídos",
    "normais": "Normais",
    "amplos": "Amplos",
    "diminuídos": "Diminuídos",
    "diminuidos": "Diminuídos",
    "outro": "Outro",
}

_MAP_B2 = {
    "normal": "Normal",
    "desdobrado": "Desdobrado",
    "fixo": "Fixo",
    "hiperfonético": "Hiperfonético", "hiperfonetico": "Hiperfonético",
    "hipofonético": "Hipofonético",   "hipofonetico": "Hipofonético",
    "hiperfonética": "Hiperfonética",
    "hiperfonetica": "Hiperfonética",
    "desdob fixo": "Desdobrado Fixo",
    "única": "Única",
    "unica": "Única",
    "outro": "Outro",
}

_MAP_SOPRO = {
    "ausente": "Ausente",
    "sistólico": "Sistólico", "sistolico": "Sistólico",
    "diastólico": "Diastólico", "diastolico": "Diastólico",
    "contínuo": "Contínuo", "continuo": "Contínuo",
    "to and fro": "To and fro",
    "sistolico e diastólico": "Sistólico e Diastólico",
    "sistolico e diastolico": "Sistólico e Diastólico",
}

_MAP_LABEL = {
    "normal": "Normal",
    "anormal": "Anormal",
    "normais": "Normal",
}


def _padronizar_col(series: pd.Series, mapa: dict) -> pd.Series:
    """Normaliza uma coluna categórica usando um mapa lower → valor canônico."""
    return series.astype(str).str.strip().str.lower().map(mapa)


def limpar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza: SEXO, PULSOS, B2, SOPRO, NORMAL_X_ANORMAL, CONVENIO,
               HDA1, HDA2, MOTIVO1, MOTIVO2.
    Valores não reconhecidos → NaN.
    """
    df["SEXO"]            = _padronizar_col(df["SEXO"],            _MAP_SEXO)
    df["PULSOS"]          = _padronizar_col(df["PULSOS"],          _MAP_PULSOS)
    df["B2"]              = _padronizar_col(df["B2"],              _MAP_B2)
    df["SOPRO"]           = _padronizar_col(df["SOPRO"],           _MAP_SOPRO)
    df["NORMAL_X_ANORMAL"] = _padronizar_col(df["NORMAL_X_ANORMAL"], _MAP_LABEL)

    # Convênio: strip + título (preservar variação natural)
    df["CONVENIO"] = df["CONVENIO"].astype(str).str.strip().str.upper().replace("NAN", np.nan)

    # PPA textual → tratar "Não Calculado" como NaN numérico
    # (o valor numérico real será gerado na etapa de feature engineering)
    df["PPA"] = pd.to_numeric(df["PPA"], errors="coerce")

    # HDA e MOTIVO: strip + título
    for col in ["HDA1", "HDA2", "MOTIVO1", "MOTIVO2"]:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan, "NaN": np.nan})
        )

    # Contagem de NaN por coluna categórica
    cats = ["SEXO", "PULSOS", "B2", "SOPRO", "NORMAL_X_ANORMAL", "CONVENIO"]
    for c in cats:
        n = df[c].isna().sum()
        print(f"[categoricas] {c}: {n} NaN após padronização.")

    return df


# ─────────────────────────────────────────────
# 5. DUPLICATAS E COMPLETUDE
# ─────────────────────────────────────────────

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com ID duplicado, mantendo a primeira ocorrência."""
    n_antes = len(df)
    df = df.drop_duplicates(subset=["ID"], keep="first")
    print(f"[duplicatas] Removidos {n_antes - len(df)} registros duplicados por ID.")
    return df


def relatorio_missings(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com contagem e % de missings por coluna."""
    total = len(df)
    miss = df.isna().sum()
    pct  = (miss / total * 100).round(2)
    rel = pd.DataFrame({"missing": miss, "pct_%": pct}).sort_values("pct_%", ascending=False)
    rel = rel[rel["missing"] > 0]
    return rel


# ─────────────────────────────────────────────
# 6. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def pipeline_limpeza(
    input_path:  Path = RAW_PATH,
    output_path: Path = PROCESSED_PATH,
    log_path:    Path = LOG_PATH,
    salvar:      bool = True,
) -> pd.DataFrame:
    """
    Executa toda a etapa de limpeza e retorna o DataFrame limpo.
    Se salvar=True, exporta para output_path e grava log em log_path.
    """
    import sys
    from io import StringIO

    # Capturar prints para log
    log_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_buffer

    print("=" * 60)
    print("PIPELINE DE LIMPEZA — UCMF/RHP")
    print("=" * 60)

    df = carregar_dataset(input_path)
    df = converter_datas(df)
    df = remover_duplicatas(df)
    df = limpar_numericas(df)
    df = limpar_categoricas(df)

    print("\n── Missings após limpeza ──")
    rel = relatorio_missings(df)
    print(rel.to_string())

    print(f"\n[pipeline] Dataset final: {len(df)} linhas × {len(df.columns)} colunas.")
    print("=" * 60)

    sys.stdout = original_stdout
    log_content = log_buffer.getvalue()
    print(log_content)  # também exibe no console

    if salvar:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[pipeline] Salvo em: {output_path}")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log_content, encoding="utf-8")
        print(f"[pipeline] Log salvo em: {log_path}")

    return df


# ─────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_clean = pipeline_limpeza()
