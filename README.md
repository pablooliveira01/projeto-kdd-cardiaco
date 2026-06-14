# Predição de Patologias Cardíacas em Crianças e Adolescentes (KDD)

## Objetivo
Aplicar o processo de KDD (Knowledge Discovery in Databases) sobre dados de
pacientes do Real Hospital Português (RHP) para predição/extração de
conhecimento sobre patologias cardíacas (NORMAL X ANORMAL).

## Estrutura do dataset
ID, Peso, Altura, IMC, Atendimento, DN, IDADE, Convenio, PULSOS, PA_SISTOLICA,
PA_DIASTOLICA, PPA, NORMAL_X_ANORMAL, B2, SOPRO, FC, HDA1, HDA2, SEXO,
MOTIVO1, MOTIVO2

## Divisão de tarefas (4 integrantes)

### Seleção e Limpeza dos Dados (Data Cleaning)
- [ ] Importar dataset e verificar tipos de dados
- [ ] Tratar valores ausentes/inconsistentes (Peso, Altura, IDADE, PA)
- [ ] Padronizar SEXO, Convenio, HDA1/HDA2, MOTIVO1/MOTIVO2 (categorias)
- [ ] Tratar/recalcular DN x Atendimento x IDADE (consistência)
- [ ] Remover/justificar outliers (ex: PA negativa, IMC absurdo)
- [ ] Documentar decisões de limpeza no notebook/markdown

### Engenharia de Atributos (Feature Engineering)
- [ ] Recalcular IMC = Peso / Altura² e comparar com IMC informado
- [ ] Classificar IMC por percentil (tabelas Portugal/CDC: normal, excesso de
  peso >p85, obesidade >p95), separado por sexo e idade
- [ ] Calcular variável PPA a partir de PA_SISTOLICA/PA_DIASTOLICA usando as
  tabelas de percentil de pressão arterial por idade/altura/sexo (anexo
  "Tables_of_blood_pressure_for_children")
- [ ] Classificar PPA (normal, pré-hipertensão, HAS estágio 1/2) — critério:
  PAS OU PAD ≥ percentil correspondente
- [ ] Criar faixas etárias clinicamente relevantes (ex: 0-2, 2-5, 6-12, 13-19)
- [ ] Documentar fórmulas e fontes (referências bibliográficas)

### Análise Exploratória (EDA) e Visualização
- [ ] Estatísticas descritivas por variável (numéricas e categóricas)
- [ ] Distribuição de NORMAL X ANORMAL (balanceamento de classes)
- [ ] Correlações entre variáveis numéricas (Peso, Altura, IMC, IDADE, PA, FC)
- [ ] Gráficos: histogramas, boxplots por sexo/grupo, dispersão IMC x IDADE,
  PA x IDADE, B2/SOPRO x diagnóstico
- [ ] Comparar grupos NORMAL vs ANORMAL (PULSOS, B2, SOPRO, HDA1/2)
- [ ] Relatório de insights preliminares

### Modelagem (Mineração de Dados) e Avaliação
- [ ] Pré-processamento final: encoding de categóricas, normalização
- [ ] Split treino/teste (e validação cruzada)
- [ ] Treinar modelos (ex: Árvore de Decisão, Random Forest, KNN, Naive Bayes,
  Regressão Logística) para prever NORMAL X ANORMAL
- [ ] Avaliar métricas (acurácia, precisão, recall, F1, matriz de confusão)
- [ ] Comparar modelos e selecionar melhor
- [ ] Interpretar atributos mais importantes (feature importance)

## Etapas transversais (todos)
- [ ] Revisão do referencial teórico/clínico (intervalos de IDADE, IMC, PA
  usados na etapa 2)
- [ ] Redação conjunta do relatório final (Introdução, Metodologia, Resultados,
  Conclusão) seguindo etapas do KDD: Seleção → Pré-processamento →
  Transformação → Mineração → Interpretação/Avaliação
- [ ] Revisão final e formatação

## Estrutura de pastas

```
projeto-kdd-cardiaco/
├── README.md
├── data/
│   ├── raw/                # dataset original, sem alterações
│   └── processed/          # dataset limpo e com features adicionais
├── docs/
│   └── referencias/        # PDFs/tabelas de apoio (IMC, PA, etc.)
├── notebooks/
│   ├── 01_limpeza.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modelagem.ipynb
├── src/
│   ├── __init__.py
│   ├── limpeza.py
│   ├── features.py
│   ├── eda.py
│   └── modelagem.py
├── reports/
│   ├── figures/             # gráficos exportados
│   └── relatorio_final.md
└── requirements.txt
```

## Fluxo de trabalho no GitHub
1. Cada integrante trabalha em uma branch própria (`feature/limpeza`,
   `feature/feature-engineering`, `feature/eda`, `feature/modelagem`)
2. Notebooks/scripts ficam em `/notebooks` ou `/src`, nomeados por etapa
3. Ao concluir uma etapa: commit, push, abrir Pull Request para `main`
4. Atualizar este README marcando o checkbox correspondente e adicionando
   nome do responsável + data
5. Reunião rápida semanal para sincronizar dependências (ex: Pessoa 2 depende
   da limpeza da Pessoa 1; Pessoa 4 depende de 1, 2 e 3)

## Dependências entre etapas
```
Limpeza (P1) → Feature Engineering (P2) → EDA (P3) → Modelagem (P4)
```
Sugestão: P1 entrega versão inicial rápido (mesmo que incompleta) para
desbloquear P2 e P3 em paralelo; ajustes incrementais via PR.

## Referências
- Tabelas de pressão arterial por idade/altura/sexo (NHBPEP/NHLBI)
- Curvas de IMC por percentil (DGS Portugal, Circular 05/DSMIA)
- Pinto Jr. et al. (2004) — incidência de cardiopatia congênita no Brasil
