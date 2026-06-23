# Roteiro da apresentação (meta: 9 minutos, limite de 10)

Slides em `reports/apresentacao.pdf` (10 telas, 16:9). Divisão sugerida para
4 integrantes, com tempo por bloco. Falem devagar; sobra cerca de 1 minuto de
folga para a transição e perguntas.

---

## Bloco 1 — Abertura · Gabriel Henrique · ~1 min

**Slide 1 (capa).** Bom dia. Somos o grupo [nº], e vamos apresentar a aplicação
do processo de KDD para prever patologias cardíacas em crianças e adolescentes,
usando dados reais do Real Hospital Português.

**Slide 2 (problema e objetivo).**
- Cardiopatias na infância muitas vezes evoluem de forma silenciosa, então
  detectá-las cedo é importante.
- Trabalhamos com uma base real de quase 13 mil atendimentos de pacientes de 0
  a 19 anos.
- O objetivo foi prever se o paciente é normal ou anormal do ponto de vista
  cardíaco, percorrendo todas as fases do KDD — e não só buscar acurácia alta.

---

## Bloco 2 — Preparação dos dados · Gabriel Nery · ~2 min

**Slide 3 (limpeza).**
- Na limpeza, a prioridade foi corrigir erros de digitação e valores
  impossíveis, sem apagar registros.
- Recalculamos sempre o IMC a partir de peso e altura; isso derrubou o valor
  máximo de 847 para 39,5, o que mostra o tamanho dos erros de digitação.
- Alturas, pressões e frequências fisiologicamente impossíveis viraram
  ausentes.
- Um ponto importante: a grande quantidade de pressões e medidas ausentes não é
  erro de coleta, é característica da base — nem todo paciente é medido. Por
  isso tratamos por imputação, não por exclusão.

**Slide 4 (transformações).**
- Em seguida, criamos cinco atributos clínicos novos: percentis e classificação
  de IMC e de estatura por idade e sexo, a classificação da pressão arterial
  segundo uma tabela oficial, e a faixa etária.
- E preparamos os dados para os modelos com imputação, padronização e
  codificação one-hot das variáveis categóricas.

---

## Bloco 3 — Exploração e modelagem · José Eduardo · ~2 min

**Slide 5 (EDA).**
- Na análise exploratória, as classes estão equilibradas, cerca de 58% normais
  e 42% anormais.
- O que mais chamou atenção é que os sinais de ausculta — sopro, bulha B2 e
  pulsos — têm relação muito forte com o diagnóstico; um sopro sistólico, por
  exemplo, aparece como anormal em cerca de 96% dos casos.
- Já peso, IMC e pressão têm relação fraca com o alvo.

**Slide 6 (modelagem).**
- Treinamos cinco modelos clássicos: árvore de decisão, random forest, KNN,
  naive bayes e regressão logística.
- Usamos o pipeline do scikit-learn para garantir que o pré-processamento fosse
  ajustado só no treino, evitando vazamento de dados; dividimos 75% para treino
  e 25% para teste, com validação cruzada.
- E, atendendo ao enunciado, avaliamos tudo em dois cenários: com e sem a
  variável sopro.

---

## Bloco 4 — Resultados e conclusão · Pablo · ~3 min

**Slide 7 (resultados com e sem sopro).**
- Este é o resultado central. Com o sopro, os melhores modelos chegam a um F1 de
  0,91. Sem o sopro, o desempenho cai para cerca de 0,55.
- E, como mostra o gráfico, a queda acontece em todos os cinco modelos. Ou seja,
  a ausculta concentra quase toda a capacidade de previsão.

**Slide 8 (melhor modelo).**
- O modelo escolhido foi a regressão logística, por ser simples e interpretável.
  Ela acerta a grande maioria dos casos, com poucos falsos positivos e falsos
  negativos.
- A análise de importância confirma a leitura clínica: o sopro domina, seguido
  da bulha B2; peso, idade e pressão pesam pouco.

**Slide 9 (conclusão e ressalvas).**
- A conclusão é coerente com a cardiologia: o diagnóstico aqui é definido
  principalmente pela ausculta.
- Mas fazemos uma ressalva honesta: como a base é de pacientes já encaminhados
  ao cardiologista, o sopro prevê tão bem em parte por viés de seleção, e a
  variável só diz o tipo de sopro, não se ele é inocente ou patológico.
- Num rastreio da população geral isso mudaria, porque a maioria dos sopros na
  infância é inocente. É por isso que apresentar as duas versões é essencial.

**Slide 10 (síntese).**
- Resumindo: percorremos todas as fases do KDD, chegamos a um bom modelo e,
  mais do que acurácia, extraímos conhecimento sobre o que pesa no diagnóstico.
- Obrigado. Estamos abertos a perguntas.

---

## Dicas rápidas
- Ensaiem uma vez com cronômetro; se passar de 9 minutos, cortem detalhes dos
  slides 3 e 5.
- Apontem para o gráfico do slide 7 ao falar a frase "cai em todos os modelos".
- Deixem números secundários (precisões por modelo) para responder se
  perguntarem; no discurso, foquem na mensagem de cada slide.
