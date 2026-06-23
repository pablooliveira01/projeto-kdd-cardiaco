<!-- class: capa -->

<p class="kicker">Mineração de Dados — Estudo de Caso</p>

# Predição de Patologias Cardíacas em Crianças e Adolescentes

<p class="sub">Aplicação do processo de KDD à base do Real Hospital Português (RHP/UCMF)</p>

<p class="meta"><strong>Integrantes</strong><br>
Gabriel Henrique Ribeiro Amâncio<br>
Gabriel Nery da Silva Espindola<br>
José Eduardo Gontijo de Carvalho<br>
Pablo Agnaldo Marques Oliveira</p>

---

## O problema e o objetivo

- Cardiopatias na infância podem evoluir de forma **silenciosa**; a detecção precoce é decisiva.
- Base **real** do Real Hospital Português (Recife): **12.873 atendimentos** de pacientes de **0 a 19 anos**.
- Objetivo: aplicar o **processo de KDD** para prever a variável-alvo — paciente **Normal** ou **Anormal** do ponto de vista cardíaco.
- Foco do trabalho: explorar **todas as fases** do KDD, e não apenas maximizar a acurácia.

---

## Etapa 1 — Limpeza dos dados

- Corrigimos erros de digitação e valores implausíveis **sem excluir registros** e preservando o arquivo original.
- O **IMC foi sempre recalculado** (peso ÷ altura²); o valor máximo caiu de **847** para **39,5**.
- Alturas abaixo de 40 cm, pressões acima de 250 e frequências fora de 30–300 bpm passaram a **ausentes**.
- A grande ausência em pressão arterial (cerca de 60%) e antropometria (cerca de 34%) é **estrutural** da base — tratada por imputação, não por exclusão.

---

## Etapa 2 — Transformações (engenharia de atributos)

- Criamos **cinco atributos clínicos** derivados (35 colunas no total):
    - Percentil e classificação de **IMC** por idade e sexo
    - Percentil de **estatura**
    - Classificação de **pressão arterial** (tabela oficial NHBPEP/NHLBI)
    - **Faixa etária** clinicamente relevante
- Preparação para a modelagem: imputação (mediana ou categoria própria), padronização e **codificação one-hot**.

---

## Etapa 3 — Exploração dos dados (EDA)

- Classes **equilibradas**: 57,6% Normal e 42,4% Anormal.
- Os sinais de **ausculta** se destacam na relação com o diagnóstico:
    - Sopro sistólico associado a Anormal em cerca de **96%** dos casos
    - Bulha B2 alterada e pulsos diminuídos também muito associados
- Peso, IMC e pressão arterial mostram relação **fraca** com o alvo.

---

## Etapa 4 — Modelagem

- Cinco classificadores: **Árvore de Decisão, Random Forest, KNN, Naive Bayes e Regressão Logística**.
- Pipeline do scikit-learn com pré-processamento ajustado **apenas no treino** (sem vazamento de informação).
- Divisão estratificada **75% treino / 25% teste** e **validação cruzada** de cinco partições.
- Conforme o enunciado, tudo foi avaliado em **dois cenários: com e sem a variável SOPRO**.
- Métricas: acurácia, precisão, recall, F1 e AUC (classe positiva = Anormal).

---

## Resultados: com e sem SOPRO

- **Com SOPRO:** os melhores modelos chegam a **F1 de 0,91** (acurácia de 0,93).
- **Sem SOPRO:** o desempenho cai para **F1 de 0,55** (acurácia de 0,66).
- A queda ocorre em **todos os cinco modelos** — a ausculta concentra quase toda a capacidade preditiva.

<figure>
<img src="figures/04_comparacao_modelos.png" alt="F1 por modelo, com e sem sopro">
<figcaption>F1 da classe Anormal por modelo: a barra vermelha (sem sopro) cai em todos os classificadores.</figcaption>
</figure>

---

## Melhor modelo e interpretação

- **Regressão Logística** (com sopro): acurácia 0,929, F1 0,913 e AUC 0,947.
- Matriz de confusão (2.927 casos): **1.632 e 1.088 acertos**, com apenas 55 falsos positivos e 152 falsos negativos.
- A importância dos atributos confirma a leitura clínica: o **sopro domina**, seguido da bulha B2; antropometria, idade e pressão têm peso marginal.

<div class="figrow">
<figure><img src="figures/04_importancia_com_sopro.png" alt="Importância com sopro"><figcaption>Com sopro: o sopro domina</figcaption></figure>
<figure><img src="figures/04_importancia_sem_sopro.png" alt="Importância sem sopro"><figcaption>Sem sopro: frequência cardíaca, B2 e idade assumem</figcaption></figure>
</div>

---

## Conclusão e ressalvas

- O diagnóstico, neste contexto, é determinado **essencialmente pela ausculta** (sopro e bulha B2), e não por peso, IMC ou pressão.
- **Ressalva importante:** a base reúne pacientes **já encaminhados** à cardiologia (viés de seleção), e a variável registra apenas o **tipo** de sopro, sem separar inocente de patológico.
- Em rastreio na população geral o quadro **mudaria**, pois a maioria dos sopros na infância é inocente.
- Por isso as **duas versões** (com e sem sopro) são essenciais: a versão sem sopro mostra o que as demais variáveis realmente acrescentam.

---

## Síntese

- Percorremos **todas as fases do KDD**: seleção, limpeza, transformação, mineração e avaliação.
- O melhor modelo prevê patologia cardíaca com **F1 de 0,91** apoiando-se sobretudo no sopro.
- Mais do que a acurácia, o trabalho **extraiu conhecimento**: identificou o que pesa no diagnóstico e os cuidados de interpretação.

<p style="margin-top:10mm; font-size:20pt; color:#11304e;"><strong>Obrigado! Perguntas?</strong></p>
