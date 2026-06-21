"""
referencias_clinicas.py — Tabelas de referência clínica para a Etapa 2 (Feature Engineering)
Projeto: Predição de Patologias Cardíacas (UCMF/RHP)

Fontes:
- Estatura por idade/sexo/percentil: curvas DGS Portugal (estatura 2-20 anos,
  comprimento 0-24 meses), digitalizadas a partir de gráfico fornecido pelo
  professor. ATENÇÃO: valores lidos visualmente do gráfico — possuem margem
  de erro de leitura (~1-2 cm). Documentar essa limitação no relatório final.
- Pressão arterial por idade/percentil de altura: NHBPEP/NHLBI — "The Fourth
  Report on the Diagnosis, Evaluation, and Treatment of High Blood Pressure
  in Children and Adolescents". Pediatrics 2004; 114(2): 555-576.
  Tabela fornecida pelo usuário (documento "Tabela 1/2 - Valores da TA por
  Idade e Percentil de Altura"), valores exatos e oficiais (não estimados).
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# 1. CURVAS DE ESTATURA POR IDADE (DGS Portugal) — 2 a 20 anos
# ═══════════════════════════════════════════════════════════════════════════
# Estrutura: {idade_anos: {percentil: estatura_cm}}
# Percentis disponíveis: 5, 10, 25, 50, 75, 90, 95
#
# AVISO METODOLÓGICO: valores extraídos visualmente do gráfico fornecido
# (curvas "estatura 2-20 anos", RAPAZES/RAPARIGAS). A leitura é uma
# aproximação — pontos foram lidos em interseções de grade a cada ~1-2 anos
# onde a curva cruza linhas inteiras de cm, e interpolados nos demais.
# Margem de erro estimada: ±1-2 cm por leitura.

ESTATURA_RAPAZES = {
    2:  {5: 81,  10: 83,  25: 86,  50: 88,  75: 91,  90: 93,  95: 95},
    3:  {5: 89,  10: 91,  25: 93,  50: 96,  75: 99,  90: 101, 95: 103},
    4:  {5: 95,  10: 97,  25: 100, 50: 103, 75: 106, 90: 109, 95: 111},
    5:  {5: 102, 10: 104, 25: 107, 50: 110, 75: 113, 90: 116, 95: 118},
    6:  {5: 107, 10: 109, 25: 113, 50: 116, 75: 119, 90: 122, 95: 125},
    7:  {5: 112, 10: 115, 25: 118, 50: 122, 75: 125, 90: 128, 95: 131},
    8:  {5: 117, 10: 120, 25: 123, 50: 127, 75: 130, 90: 134, 95: 136},
    9:  {5: 121, 10: 124, 25: 128, 50: 132, 75: 135, 90: 139, 95: 142},
    10: {5: 125, 10: 128, 25: 132, 50: 137, 75: 140, 90: 144, 95: 147},
    11: {5: 129, 10: 132, 25: 137, 50: 142, 75: 146, 90: 150, 95: 153},
    12: {5: 133, 10: 137, 25: 142, 50: 148, 75: 152, 90: 157, 95: 160},
    13: {5: 139, 10: 143, 25: 149, 50: 155, 75: 160, 90: 165, 95: 168},
    14: {5: 146, 10: 150, 25: 156, 50: 162, 75: 167, 90: 172, 95: 175},
    15: {5: 153, 10: 157, 25: 162, 50: 168, 75: 173, 90: 177, 95: 180},
    16: {5: 158, 10: 162, 25: 167, 50: 172, 75: 177, 90: 181, 95: 184},
    17: {5: 161, 10: 165, 25: 170, 50: 175, 75: 179, 90: 183, 95: 186},
    18: {5: 163, 10: 166, 25: 171, 50: 176, 75: 180, 90: 184, 95: 187},
    19: {5: 164, 10: 167, 25: 172, 50: 177, 75: 181, 90: 185, 95: 188},
    20: {5: 165, 10: 168, 25: 173, 50: 178, 75: 182, 90: 186, 95: 189},
}

ESTATURA_RAPARIGAS = {
    2:  {5: 80,  10: 82,  25: 85,  50: 87,  75: 90,  90: 92,  95: 94},
    3:  {5: 88,  10: 90,  25: 93,  50: 96,  75: 98,  90: 101, 95: 103},
    4:  {5: 95,  10: 97,  25: 100, 50: 103, 75: 105, 90: 108, 95: 110},
    5:  {5: 101, 10: 103, 25: 106, 50: 109, 75: 112, 90: 115, 95: 117},
    6:  {5: 107, 10: 109, 25: 112, 50: 115, 75: 118, 90: 121, 95: 124},
    7:  {5: 112, 10: 114, 25: 117, 50: 121, 75: 124, 90: 127, 95: 130},
    8:  {5: 116, 10: 119, 25: 122, 50: 126, 75: 130, 90: 133, 95: 136},
    9:  {5: 121, 10: 124, 25: 127, 50: 131, 75: 135, 90: 139, 95: 142},
    10: {5: 125, 10: 128, 25: 132, 50: 137, 75: 141, 90: 145, 95: 148},
    11: {5: 130, 10: 133, 25: 138, 50: 142, 75: 147, 90: 151, 95: 154},
    12: {5: 136, 10: 139, 25: 144, 50: 149, 75: 153, 90: 157, 95: 160},
    13: {5: 142, 10: 145, 25: 149, 50: 154, 75: 158, 90: 162, 95: 164},
    14: {5: 146, 10: 149, 25: 153, 50: 157, 75: 161, 90: 164, 95: 167},
    15: {5: 149, 10: 151, 25: 155, 50: 159, 75: 162, 90: 166, 95: 168},
    16: {5: 150, 10: 152, 25: 156, 50: 160, 75: 163, 90: 166, 95: 169},
    17: {5: 151, 10: 153, 25: 156, 50: 160, 75: 163, 90: 167, 95: 169},
    18: {5: 151, 10: 153, 25: 157, 50: 160, 75: 164, 90: 167, 95: 170},
    19: {5: 152, 10: 154, 25: 157, 50: 161, 75: 164, 90: 167, 95: 170},
    20: {5: 152, 10: 154, 25: 157, 50: 161, 75: 164, 90: 167, 95: 170},
}

_PERCENTIS_DISPONIVEIS = [5, 10, 25, 50, 75, 90, 95]


# ═══════════════════════════════════════════════════════════════════════════
# 1B. CURVAS DE IMC POR IDADE (DGS Portugal) — 2 a 20 anos
# ═══════════════════════════════════════════════════════════════════════════
# Estrutura: {idade_anos: {percentil: imc_kg_m2}}
# AVISO METODOLÓGICO: valores extraídos visualmente do gráfico "índice de
# massa corporal 2-20 anos" (RAPAZES/RAPARIGAS) fornecido pelo professor.
# Mesma ressalva de margem de erro de leitura (~0.5 kg/m²) da tabela de
# estatura. Cortes oficiais confirmados pelo material do professor:
#   Obesidade     > percentil 95
#   Excesso de peso > percentil 85 e < percentil 95

IMC_RAPAZES = {
    2:  {5: 14.8, 10: 15.2, 25: 15.9, 50: 16.6, 75: 17.3, 90: 18.0, 95: 18.4},
    3:  {5: 14.0, 10: 14.4, 25: 15.0, 50: 15.7, 75: 16.4, 90: 17.1, 95: 17.6},
    4:  {5: 13.6, 10: 14.0, 25: 14.5, 50: 15.2, 75: 15.9, 90: 16.7, 95: 17.3},
    5:  {5: 13.4, 10: 13.7, 25: 14.3, 50: 15.0, 75: 15.7, 90: 16.6, 95: 17.4},
    6:  {5: 13.3, 10: 13.6, 25: 14.2, 50: 14.9, 75: 15.7, 90: 16.7, 95: 17.6},
    7:  {5: 13.4, 10: 13.7, 25: 14.3, 50: 15.0, 75: 15.9, 90: 17.0, 95: 18.0},
    8:  {5: 13.6, 10: 13.9, 25: 14.5, 50: 15.3, 75: 16.3, 90: 17.5, 95: 18.6},
    9:  {5: 13.8, 10: 14.2, 25: 14.8, 50: 15.7, 75: 16.8, 90: 18.1, 95: 19.3},
    10: {5: 14.1, 10: 14.5, 25: 15.2, 50: 16.1, 75: 17.4, 90: 18.8, 95: 20.1},
    11: {5: 14.4, 10: 14.9, 25: 15.7, 50: 16.7, 75: 18.1, 90: 19.6, 95: 21.0},
    12: {5: 14.8, 10: 15.3, 25: 16.2, 50: 17.4, 75: 18.8, 90: 20.4, 95: 21.9},
    13: {5: 15.2, 10: 15.8, 25: 16.8, 50: 18.1, 75: 19.6, 90: 21.3, 95: 22.8},
    14: {5: 15.7, 10: 16.3, 25: 17.4, 50: 18.8, 75: 20.3, 90: 22.0, 95: 23.7},
    15: {5: 16.2, 10: 16.9, 25: 18.0, 50: 19.4, 75: 21.0, 90: 22.7, 95: 24.5},
    16: {5: 16.7, 10: 17.4, 25: 18.6, 50: 20.0, 75: 21.6, 90: 23.4, 95: 25.2},
    17: {5: 17.2, 10: 17.9, 25: 19.1, 50: 20.6, 75: 22.2, 90: 24.0, 95: 25.9},
    18: {5: 17.6, 10: 18.3, 25: 19.6, 50: 21.1, 75: 22.8, 90: 24.6, 95: 26.5},
    19: {5: 18.0, 10: 18.7, 25: 20.0, 50: 21.6, 75: 23.3, 90: 25.2, 95: 27.1},
    20: {5: 18.3, 10: 19.0, 25: 20.4, 50: 22.0, 75: 23.8, 90: 25.7, 95: 27.6},
}

IMC_RAPARIGAS = {
    2:  {5: 14.5, 10: 14.9, 25: 15.6, 50: 16.4, 75: 17.2, 90: 18.0, 95: 18.6},
    3:  {5: 13.8, 10: 14.1, 25: 14.7, 50: 15.4, 75: 16.2, 90: 17.0, 95: 17.6},
    4:  {5: 13.3, 10: 13.6, 25: 14.2, 50: 14.9, 75: 15.7, 90: 16.5, 95: 17.2},
    5:  {5: 12.9, 10: 13.3, 25: 13.8, 50: 14.5, 75: 15.3, 90: 16.3, 95: 17.1},
    6:  {5: 12.8, 10: 13.1, 25: 13.7, 50: 14.4, 75: 15.3, 90: 16.4, 95: 17.4},
    7:  {5: 12.8, 10: 13.1, 25: 13.7, 50: 14.5, 75: 15.5, 90: 16.8, 95: 18.0},
    8:  {5: 12.9, 10: 13.2, 25: 13.9, 50: 14.8, 75: 15.9, 90: 17.3, 95: 18.7},
    9:  {5: 13.1, 10: 13.5, 25: 14.2, 50: 15.2, 75: 16.5, 90: 18.0, 95: 19.6},
    10: {5: 13.4, 10: 13.8, 25: 14.6, 50: 15.7, 75: 17.1, 90: 18.8, 95: 20.5},
    11: {5: 13.8, 10: 14.2, 25: 15.1, 50: 16.3, 75: 17.9, 90: 19.6, 95: 21.5},
    12: {5: 14.2, 10: 14.7, 25: 15.7, 50: 17.0, 75: 18.6, 90: 20.5, 95: 22.4},
    13: {5: 14.7, 10: 15.2, 25: 16.3, 50: 17.7, 75: 19.4, 90: 21.3, 95: 23.3},
    14: {5: 15.2, 10: 15.8, 25: 16.9, 50: 18.3, 75: 20.1, 90: 22.1, 95: 24.2},
    15: {5: 15.7, 10: 16.3, 25: 17.5, 50: 19.0, 75: 20.8, 90: 22.8, 95: 25.0},
    16: {5: 16.1, 10: 16.8, 25: 18.0, 50: 19.5, 75: 21.4, 90: 23.5, 95: 25.7},
    17: {5: 16.5, 10: 17.2, 25: 18.5, 50: 20.0, 75: 22.0, 90: 24.1, 95: 26.4},
    18: {5: 16.8, 10: 17.6, 25: 18.9, 50: 20.5, 75: 22.5, 90: 24.7, 95: 27.0},
    19: {5: 17.1, 10: 17.9, 25: 19.3, 50: 20.9, 75: 23.0, 90: 25.2, 95: 27.6},
    20: {5: 17.4, 10: 18.2, 25: 19.6, 50: 21.3, 75: 23.4, 90: 25.7, 95: 28.1},
}


def percentil_imc(idade: float, imc: float, sexo: str) -> float:
    """
    Estima o percentil de IMC (interpolado, escala contínua 0-100) por
    idade e sexo, usando as curvas DGS digitalizadas visualmente.

    Retorna np.nan se idade fora de 2-20 anos, sexo inválido, ou IMC ausente.
    Mesma metodologia de interpolação/extrapolação de percentil_altura().
    """
    if pd.isna(idade) or pd.isna(imc) or sexo not in ("M", "F"):
        return np.nan

    idade_r = int(round(idade))
    idade_r = max(2, min(20, idade_r))

    tabela = IMC_RAPAZES if sexo == "M" else IMC_RAPARIGAS
    if idade_r not in tabela:
        return np.nan

    pontos = tabela[idade_r]
    percentis = _PERCENTIS_DISPONIVEIS
    valores = [pontos[p] for p in percentis]

    perc_estimado = np.interp(
        imc, valores, percentis,
        left=max(1, percentis[0] - (valores[0] - imc) * 0.5),
        right=min(99, percentis[-1] + (imc - valores[-1]) * 0.5),
    )
    return round(float(np.clip(perc_estimado, 1, 99)), 1)


def classificar_imc_percentil(idade: float, imc: float, sexo: str) -> str:
    """
    Classifica o IMC por percentil de idade/sexo, conforme cortes
    oficiais confirmados pelo professor:
      < P85           → Eutrófico (peso normal)
      P85 – P95       → Excesso de peso
      >= P95          → Obesidade
      (sem corte de baixo peso explícito no material — P< 5 tratado como
       "Baixo peso", convenção pediátrica padrão)

    LIMITAÇÃO DOCUMENTADA: as curvas DGS disponíveis cobrem apenas 2-20
    anos. Bebês com idade < 2 anos (faixa "0-2 anos" do projeto) retornam
    "Não Calculado" mesmo com IMC válido, pois exigiriam uma curva de
    referência diferente (peso/comprimento por idade, 0-24 meses), não
    disponibilizada para este trabalho. Da mesma forma, SEXO="I"
    (indeterminado) não possui curva própria e também retorna
    "Não Calculado".
    """
    if idade is not None and not pd.isna(idade) and idade < 2:
        return "Não Calculado"

    p = percentil_imc(idade, imc, sexo)
    if pd.isna(p):
        return "Não Calculado"
    if p < 5:
        return "Baixo peso"
    elif p < 85:
        return "Eutrófico"
    elif p < 95:
        return "Excesso de peso"
    else:
        return "Obesidade"


def percentil_altura(idade: float, altura_cm: float, sexo: str) -> float:
    """
    Estima o percentil de altura (interpolado, escala contínua 0-100)
    de uma criança/adolescente dada idade (anos), altura (cm) e sexo ('M'/'F').

    Retorna np.nan se idade fora de 2-20 anos, sexo inválido, ou altura ausente.

    Método:
    1. Idade é arredondada para o inteiro mais próximo disponível na tabela
       (tabelas DGS são por ano completo).
    2. Para essa idade, interpola linearmente a altura informada entre os
       pontos de percentil conhecidos (5,10,25,50,75,90,95) para obter um
       percentil contínuo aproximado.
    3. Extrapola levemente além de P5/P95 mantendo a inclinação do último
       segmento (clipado em [1, 99] para evitar valores absurdos).
    """
    if pd.isna(idade) or pd.isna(altura_cm) or sexo not in ("M", "F"):
        return np.nan

    idade_r = int(round(idade))
    idade_r = max(2, min(20, idade_r))

    tabela = ESTATURA_RAPAZES if sexo == "M" else ESTATURA_RAPARIGAS
    if idade_r not in tabela:
        return np.nan

    pontos = tabela[idade_r]
    percentis = _PERCENTIS_DISPONIVEIS
    alturas = [pontos[p] for p in percentis]

    # Interpolação/extrapolação linear altura → percentil
    perc_estimado = np.interp(
        altura_cm, alturas, percentis,
        left=max(1, percentis[0] - (alturas[0] - altura_cm) * 0.5),
        right=min(99, percentis[-1] + (altura_cm - alturas[-1]) * 0.5),
    )
    return round(float(np.clip(perc_estimado, 1, 99)), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TABELAS DE PRESSÃO ARTERIAL POR IDADE E PERCENTIL DE ALTURA (NHBPEP/NHLBI)
# ═══════════════════════════════════════════════════════════════════════════
# Fonte: Tabela 1 (RAPAZES) e Tabela 2 (RAPARIGAS) fornecidas pelo usuário.
# Valores EXATOS e oficiais — não estimados.
# Estrutura: {idade: {percentil_TA: {"sis": [p5..p95], "dia": [p5..p95]}}}
# Ordem dos valores em "sis"/"dia": percentis de altura [5,10,25,50,75,90,95]

PA_RAPAZES = {
    1:  {90: {"sis": [94,95,97,99,100,102,103],  "dia": [49,50,51,52,53,53,54]},
         95: {"sis": [98,99,101,103,104,106,106],"dia": [54,54,55,56,57,58,58]},
         99: {"sis": [105,106,108,110,112,113,114],"dia": [61,62,63,64,65,66,66]}},
    2:  {90: {"sis": [97,99,100,102,104,105,106], "dia": [54,55,56,57,58,58,59]},
         95: {"sis": [101,102,104,106,108,109,110],"dia": [59,59,60,61,62,63,63]},
         99: {"sis": [109,110,111,113,115,117,117],"dia": [66,67,68,69,70,71,71]}},
    3:  {90: {"sis": [100,101,103,105,107,108,109],"dia": [59,59,60,61,62,63,63]},
         95: {"sis": [104,105,107,109,110,112,113],"dia": [63,63,64,65,66,67,67]},
         99: {"sis": [111,112,114,116,118,119,120],"dia": [71,71,72,73,74,75,75]}},
    4:  {90: {"sis": [102,103,105,107,109,110,111],"dia": [62,63,64,65,66,66,67]},
         95: {"sis": [106,107,109,111,112,114,115],"dia": [66,67,68,69,70,71,71]},
         99: {"sis": [113,114,116,118,120,121,122],"dia": [74,75,76,77,78,78,79]}},
    5:  {90: {"sis": [104,105,106,108,110,111,112],"dia": [65,66,67,68,69,69,70]},
         95: {"sis": [108,109,110,112,114,115,116],"dia": [69,70,71,72,73,74,74]},
         99: {"sis": [115,116,118,120,121,123,123],"dia": [77,78,79,80,81,81,82]}},
    6:  {90: {"sis": [105,106,108,110,111,113,113],"dia": [68,68,69,70,71,72,72]},
         95: {"sis": [109,110,112,114,115,117,117],"dia": [72,72,73,74,75,76,76]},
         99: {"sis": [116,117,119,121,123,124,125],"dia": [80,80,81,82,83,84,84]}},
    7:  {90: {"sis": [106,107,109,111,113,114,115],"dia": [70,70,71,72,73,74,74]},
         95: {"sis": [110,111,113,115,117,118,119],"dia": [74,74,75,76,77,78,78]},
         99: {"sis": [117,118,120,122,124,125,126],"dia": [82,82,83,84,85,86,86]}},
    8:  {90: {"sis": [107,109,110,112,114,115,116],"dia": [71,72,72,73,74,75,76]},
         95: {"sis": [111,112,114,116,118,119,120],"dia": [75,76,77,78,79,79,80]},
         99: {"sis": [119,120,122,123,125,127,127],"dia": [83,84,85,86,87,87,88]}},
    9:  {90: {"sis": [109,110,112,114,115,117,118],"dia": [72,73,74,75,76,76,77]},
         95: {"sis": [113,114,116,118,119,121,121],"dia": [76,77,78,79,80,81,81]},
         99: {"sis": [120,121,123,125,127,128,129],"dia": [84,85,86,87,88,88,89]}},
    10: {90: {"sis": [109,110,112,114,115,117,118], "dia": [72,73,73,74,75,76,77]},
         95: {"sis": [113,114,116,118,119,121,122], "dia": [76,77,77,78,79,80,81]},
         99: {"sis": [120,121,123,125,127,128,130], "dia": [84,85,86,86,88,88,89]}},
    11: {90: {"sis": [113,114,115,117,119,120,121], "dia": [74,74,75,76,77,78,78]},
         95: {"sis": [117,118,119,121,123,124,125], "dia": [78,78,79,80,81,82,82]},
         99: {"sis": [124,125,127,129,130,132,132], "dia": [86,86,87,88,89,90,90]}},
    12: {90: {"sis": [115,116,118,120,121,123,123], "dia": [74,75,75,76,77,78,79]},
         95: {"sis": [119,120,122,123,125,127,127], "dia": [78,79,80,81,82,82,83]},
         99: {"sis": [126,127,129,131,133,134,135], "dia": [86,87,88,89,90,90,91]}},
    13: {90: {"sis": [117,118,120,122,124,125,126], "dia": [75,75,76,77,78,79,79]},
         95: {"sis": [121,122,124,126,128,129,130], "dia": [79,79,80,81,82,83,83]},
         99: {"sis": [128,130,131,133,135,136,137], "dia": [87,87,88,89,90,91,91]}},
    14: {90: {"sis": [120,121,123,125,126,128,128], "dia": [75,76,77,78,79,79,80]},
         95: {"sis": [124,125,127,128,130,132,132], "dia": [80,80,81,82,83,84,84]},
         99: {"sis": [131,132,134,136,138,139,140], "dia": [87,88,89,90,91,92,92]}},
    15: {90: {"sis": [122,124,125,127,129,130,131], "dia": [76,77,78,79,80,80,81]},
         95: {"sis": [126,127,129,131,133,134,135], "dia": [81,81,82,83,84,85,85]},
         99: {"sis": [134,135,136,138,140,142,142], "dia": [88,89,90,91,92,93,93]}},
    16: {90: {"sis": [125,126,128,130,131,133,134], "dia": [78,78,79,80,81,82,82]},
         95: {"sis": [129,130,132,134,135,137,137], "dia": [82,83,83,84,85,86,87]},
         99: {"sis": [136,137,139,141,143,144,145], "dia": [90,90,91,92,93,94,94]}},
    17: {90: {"sis": [127,128,130,132,134,135,136], "dia": [80,80,81,82,83,84,84]},
         95: {"sis": [131,132,134,136,138,139,140], "dia": [84,85,86,87,87,88,89]},
         99: {"sis": [139,140,141,143,145,146,147], "dia": [92,93,93,94,95,96,97]}},
}

PA_RAPARIGAS = {
    1:  {90: {"sis": [97,97,98,100,101,102,103],  "dia": [52,53,53,54,55,55,56]},
         95: {"sis": [100,101,102,104,105,106,107],"dia": [56,57,57,58,59,59,60]},
         99: {"sis": [108,108,109,111,112,113,114],"dia": [64,64,65,65,66,67,67]}},
    2:  {90: {"sis": [98,99,100,101,103,104,105], "dia": [57,58,58,59,60,61,61]},
         95: {"sis": [102,103,104,105,107,108,109],"dia": [61,62,62,63,64,65,65]},
         99: {"sis": [109,110,111,112,114,115,116],"dia": [69,69,70,70,71,72,72]}},
    3:  {90: {"sis": [100,100,102,103,104,106,106],"dia": [61,62,62,63,64,64,65]},
         95: {"sis": [104,104,105,107,108,109,110],"dia": [65,66,66,67,68,68,69]},
         99: {"sis": [111,111,113,114,115,116,117],"dia": [73,73,74,74,75,76,76]}},
    4:  {90: {"sis": [101,102,103,104,106,107,108],"dia": [64,64,65,66,67,67,68]},
         95: {"sis": [105,106,107,108,110,111,112],"dia": [68,68,69,70,71,71,72]},
         99: {"sis": [112,113,114,115,117,118,119],"dia": [76,76,76,77,78,79,79]}},
    5:  {90: {"sis": [103,103,105,106,107,109,109],"dia": [66,67,67,68,69,69,70]},
         95: {"sis": [107,107,108,110,111,112,113],"dia": [70,71,71,72,73,73,74]},
         99: {"sis": [114,114,116,117,118,120,120],"dia": [78,78,79,79,80,81,81]}},
    6:  {90: {"sis": [104,105,106,108,109,110,111],"dia": [68,68,69,70,70,71,72]},
         95: {"sis": [108,109,110,111,113,114,115],"dia": [72,72,73,74,74,75,76]},
         99: {"sis": [115,116,117,119,120,121,122],"dia": [80,80,80,81,82,83,83]}},
    7:  {90: {"sis": [106,107,108,109,111,112,113],"dia": [69,70,70,71,72,72,73]},
         95: {"sis": [110,111,112,113,115,116,116],"dia": [73,74,74,75,76,76,77]},
         99: {"sis": [117,118,119,120,122,123,124],"dia": [81,81,82,82,83,84,84]}},
    8:  {90: {"sis": [108,109,110,111,113,114,114],"dia": [71,71,71,72,73,74,74]},
         95: {"sis": [112,112,114,115,116,118,118],"dia": [75,75,75,76,77,78,78]},
         99: {"sis": [119,120,121,122,123,125,125],"dia": [82,82,83,83,84,85,86]}},
    9:  {90: {"sis": [110,110,112,113,114,116,116],"dia": [72,72,72,73,74,75,75]},
         95: {"sis": [114,114,115,117,118,119,120],"dia": [76,76,76,77,78,79,79]},
         99: {"sis": [121,121,123,124,125,127,127],"dia": [83,83,84,84,85,86,87]}},
    10: {90: {"sis": [112,112,114,115,116,118,118], "dia": [73,73,73,74,75,76,76]},
         95: {"sis": [116,116,117,119,120,121,122], "dia": [77,77,77,78,79,80,80]},
         99: {"sis": [123,123,125,126,127,129,129], "dia": [84,84,85,86,86,87,88]}},
    11: {90: {"sis": [114,114,116,117,118,119,120], "dia": [74,74,74,75,76,77,77]},
         95: {"sis": [118,118,119,121,122,123,124], "dia": [78,78,78,79,80,81,81]},
         99: {"sis": [125,125,126,128,129,130,131], "dia": [85,85,86,87,87,88,89]}},
    12: {90: {"sis": [116,116,117,119,120,121,122], "dia": [75,75,75,76,77,78,78]},
         95: {"sis": [119,120,121,123,124,125,126], "dia": [79,79,79,80,81,82,82]},
         99: {"sis": [127,127,128,130,131,132,133], "dia": [86,86,87,88,88,89,90]}},
    13: {90: {"sis": [117,118,119,121,122,123,124], "dia": [76,76,76,77,78,79,79]},
         95: {"sis": [121,122,123,124,126,127,128], "dia": [80,80,80,81,82,83,83]},
         99: {"sis": [128,129,130,132,133,134,135], "dia": [87,87,88,89,89,90,91]}},
    14: {90: {"sis": [119,120,121,122,124,125,125], "dia": [77,77,77,78,79,80,80]},
         95: {"sis": [123,123,125,126,127,129,129], "dia": [81,81,81,82,83,84,84]},
         99: {"sis": [130,131,132,133,135,136,136], "dia": [88,88,89,90,90,91,92]}},
    15: {90: {"sis": [120,121,122,123,125,126,127], "dia": [78,78,78,79,80,81,81]},
         95: {"sis": [124,125,126,127,129,130,131], "dia": [82,82,82,83,84,85,85]},
         99: {"sis": [131,132,133,134,136,137,138], "dia": [89,89,90,91,91,92,93]}},
    16: {90: {"sis": [121,122,123,124,126,127,128], "dia": [78,78,79,80,81,81,82]},
         95: {"sis": [125,126,127,128,130,131,132], "dia": [82,82,83,84,85,85,86]},
         99: {"sis": [132,133,134,135,137,138,139], "dia": [90,90,90,91,92,93,93]}},
    17: {90: {"sis": [122,122,123,125,126,127,128], "dia": [78,79,79,80,81,81,82]},
         95: {"sis": [125,126,127,129,130,131,132], "dia": [82,83,83,84,85,85,86]},
         99: {"sis": [133,133,134,136,137,138,139], "dia": [90,90,91,91,92,93,93]}},
}

_PERCENTIS_ALTURA_PA = [5, 10, 25, 50, 75, 90, 95]
_PERCENTIS_TA_DISPONIVEIS = [90, 95, 99]


def limite_pa(idade: float, sexo: str, percentil_altura_crianca: float, percentil_ta: int):
    """
    Retorna (limite_sistolica, limite_diastolica) em mmHg para um dado
    percentil de TA (90, 95 ou 99), interpolando entre as colunas de
    percentil de altura conforme o percentil_altura_crianca informado.

    Retorna (np.nan, np.nan) se idade fora de 1-17 anos, sexo inválido,
    ou percentil_ta não suportado.
    """
    if pd.isna(idade) or sexo not in ("M", "F") or pd.isna(percentil_altura_crianca):
        return np.nan, np.nan
    if percentil_ta not in _PERCENTIS_TA_DISPONIVEIS:
        return np.nan, np.nan

    idade_r = int(round(idade))
    idade_r = max(1, min(17, idade_r))

    tabela = PA_RAPAZES if sexo == "M" else PA_RAPARIGAS
    if idade_r not in tabela:
        return np.nan, np.nan

    linha = tabela[idade_r][percentil_ta]
    sis = np.interp(percentil_altura_crianca, _PERCENTIS_ALTURA_PA, linha["sis"])
    dia = np.interp(percentil_altura_crianca, _PERCENTIS_ALTURA_PA, linha["dia"])
    return round(float(sis), 1), round(float(dia), 1)


def classificar_ppa(pa_sistolica: float, pa_diastolica: float,
                     idade: float, sexo: str, altura_cm: float) -> str:
    """
    Classifica a Pressão Arterial (PPA) segundo NHBPEP/NHLBI:
      - "Normal"          : PAS e PAD < percentil 90
      - "Pré-hipertensão" : PAS ou PAD entre percentil 90 e 95
      - "HAS estágio 1"   : PAS ou PAD entre percentil 95 e 99+5mmHg
      - "HAS estágio 2"   : PAS ou PAD >= percentil 99 + 5mmHg
      - "Não Calculado"   : dados insuficientes (idade fora de 1-17, sexo
                            indeterminado, ou PA/altura ausente)

    Critério oficial: usa-se o MAIOR entre a classificação da sistólica
    e da diastólica (PAS OU PAD ≥ percentil correspondente).
    """
    if pd.isna(pa_sistolica) or pd.isna(pa_diastolica) or pd.isna(idade) \
            or pd.isna(altura_cm) or sexo not in ("M", "F"):
        return "Não Calculado"

    idade_r = round(idade)
    if idade_r < 1 or idade_r > 17:
        return "Não Calculado"

    p_altura = percentil_altura(idade, altura_cm, sexo)
    if pd.isna(p_altura):
        return "Não Calculado"

    p90_sis, p90_dia = limite_pa(idade, sexo, p_altura, 90)
    p95_sis, p95_dia = limite_pa(idade, sexo, p_altura, 95)
    p99_sis, p99_dia = limite_pa(idade, sexo, p_altura, 99)

    if pd.isna(p90_sis):
        return "Não Calculado"

    # HAS estágio 2: PAS ou PAD >= P99 + 5 mmHg
    if pa_sistolica >= p99_sis + 5 or pa_diastolica >= p99_dia + 5:
        return "HAS estágio 2"

    # HAS estágio 1: PAS ou PAD entre P95 e P99+5
    if pa_sistolica >= p95_sis or pa_diastolica >= p95_dia:
        return "HAS estágio 1"

    # Pré-hipertensão: PAS ou PAD entre P90 e P95
    if pa_sistolica >= p90_sis or pa_diastolica >= p90_dia:
        return "Pré-hipertensão"

    return "Normal"
