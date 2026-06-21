"""
features.py — Etapa 2 do KDD: Engenharia de Atributos (Feature Engineering)
Projeto: Predição de Patologias Cardíacas (UCMF/RHP)

Depende de: data/processed/UCMF_clean.csv (saída da Etapa 1 — limpeza.py)
Depende de: referencias_clinicas.py (tabelas de estatura e PA)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from referencias_clinicas import (
    percentil_altura,
    percentil_imc,
    classificar_ppa,
    classificar_imc_percentil,
)

CLEAN_PATH    = Path("data/processed/UCMF_clean.csv")
FEATURES_PATH = Path("data/processed/UCMF_features.csv")


# ─────────────────────────────────────────────
# 1. IMC POR PERCENTIL (classificação nutricional)
# ─────────────────────────────────────────────
# Classificação por percentil de IMC por idade/sexo, conforme curvas DGS
# (gráfico "índice de massa corporal 2-20 anos") e cortes oficiais
# confirmados pelo material do professor:
#   < P85           → Eutrófico (peso normal)
#   P85 – P95       → Excesso de peso
#   >= P95          → Obesidade
# Ver referencias_clinicas.classificar_imc_percentil() para a implementação
# e a documentação da margem de erro de leitura das curvas.


def adicionar_features_imc(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona PERCENTIL_IMC e IMC_CLASSIFICACAO ao DataFrame."""
    df["PERCENTIL_IMC"] = df.apply(
        lambda r: percentil_imc(r["IDADE"], r["IMC"], r["SEXO"]), axis=1
    )
    df["IMC_CLASSIFICACAO"] = df.apply(
        lambda r: classificar_imc_percentil(r["IDADE"], r["IMC"], r["SEXO"]), axis=1
    )
    print("[features] PERCENTIL_IMC e IMC_CLASSIFICACAO adicionadas.")
    print(df["IMC_CLASSIFICACAO"].value_counts(dropna=False).to_string())
    return df


# ─────────────────────────────────────────────
# 2. PERCENTIL DE ALTURA E PPA
# ─────────────────────────────────────────────

def adicionar_features_pa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona:
      - PERCENTIL_ALTURA: percentil estimado de estatura por idade/sexo
      - PPA_CALCULADO: classificação de PA (Normal / Pré-hipertensão /
        HAS estágio 1 / HAS estágio 2 / Não Calculado)

    Usa as tabelas em referencias_clinicas.py (NHBPEP/NHLBI + curvas DGS).
    """
    df["PERCENTIL_ALTURA"] = df.apply(
        lambda r: percentil_altura(r["IDADE"], r["ALTURA"], r["SEXO"]), axis=1
    )

    df["PPA_CALCULADO"] = df.apply(
        lambda r: classificar_ppa(
            r["PA_SISTOLICA"], r["PA_DIASTOLICA"],
            r["IDADE"], r["SEXO"], r["ALTURA"]
        ), axis=1
    )

    print("[features] PERCENTIL_ALTURA e PPA_CALCULADO adicionados.")
    print(df["PPA_CALCULADO"].value_counts(dropna=False).to_string())
    return df


# ─────────────────────────────────────────────
# 3. FAIXAS ETÁRIAS CLINICAMENTE RELEVANTES
# ─────────────────────────────────────────────

_FAIXAS_ETARIAS = [
    (0,  2,  "0-2 anos (lactente/primeira infância)"),
    (2,  5,  "2-5 anos (pré-escolar)"),
    (5,  12, "5-12 anos (escolar)"),
    (12, 19, "12-19 anos (adolescente)"),
    (19, 200, "19+ anos (fora do escopo pediátrico)"),
]


def classificar_faixa_etaria(idade: float) -> str:
    if pd.isna(idade):
        return "Não Calculado"
    for inicio, fim, label in _FAIXAS_ETARIAS:
        if inicio <= idade < fim:
            return label
    return "Não Calculado"


def adicionar_faixa_etaria(df: pd.DataFrame) -> pd.DataFrame:
    df["FAIXA_ETARIA"] = df["IDADE"].apply(classificar_faixa_etaria)
    print("[features] FAIXA_ETARIA adicionada.")
    print(df["FAIXA_ETARIA"].value_counts(dropna=False).to_string())
    return df


# ─────────────────────────────────────────────
# 4. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def pipeline_features(
    input_path:  Path = CLEAN_PATH,
    output_path: Path = FEATURES_PATH,
    salvar:      bool = True,
) -> pd.DataFrame:
    print("=" * 60)
    print("PIPELINE DE FEATURE ENGINEERING — UCMF/RHP")
    print("=" * 60)

    df = pd.read_csv(input_path, low_memory=False)
    print(f"[pipeline] Dataset carregado: {df.shape}")

    df = adicionar_features_imc(df)
    df = adicionar_features_pa(df)
    df = adicionar_faixa_etaria(df)

    print(f"\n[pipeline] Dataset final: {df.shape}")
    print("=" * 60)

    if salvar:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[pipeline] Salvo em: {output_path}")

    return df


if __name__ == "__main__":
    df_features = pipeline_features()
