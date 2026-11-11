# Experimento: WSP como mecanismo de otimização simbólica

## A pergunta de pesquisa

Dada uma coleção de observações numéricas, podemos modificar diretamente os pesos
de um modelo de linguagem para que ele proponha expressões matemáticas melhores?
E, se isso funcionar em alguns exemplos, a mudança se transfere para funções que
não participaram da escolha da intervenção?

O objetivo imediato é construir um experimento capaz de responder a essas perguntas,
inclusive quando a resposta for negativa. Não pressupomos que um agente capaz de
inspecionar pesos consiga criar uma capacidade de otimização simbólica.

Este documento descreve o experimento pretendido. Os números sugeridos para o
piloto são decisões de protocolo a registrar antes da execução, não resultados
obtidos nem configurações já impostas pelo código.

## De onde vem a ideia

O problema geral é encontrar uma expressão válida que minimize uma função objetivo:

\[
e^\star=\arg\min_{e\in\mathrm{Exp}_G}f(e).
\]

A gramática \(G\) delimita o espaço de expressões. A função \(f\) pode medir erro de
predição, complexidade ou desempenho de um programa em uma especificação.

Na proposta de 2021, a ideia era aprender um encoder, um decoder e um estimador do
objetivo. A busca ocorreria em uma representação contínua, com posterior decodificação
para expressões. Construir um decoder de árvores confiável era parte importante da
dificuldade prática.

A proposta posterior de regressão simbólica usava um LLM como gerador, um adaptador
para observações contínuas e PPO para ajustar seu comportamento. Ela reduz a necessidade
de construir um decoder próprio, mas exige operar um pipeline de treinamento.

O experimento atual investiga outra possibilidade: usar um agente para formular e
testar intervenções diretas nos pesos de um gerador pré-treinado. Em vez de atualizar
parâmetros por gradientes, aplicamos uma modificação explícita:

\[
\theta' = \theta + \Delta\theta_{\mathrm{agente}}.
\]

O agente pode inspecionar tensores, observar ativações e medir comportamento. Ele não
faz backpropagation, PPO ou fine-tuning do gerador. As observações entram como texto;
não há PointAdapter nem alteração do tokenizer ou da cabeça de vocabulário.

Uma melhora nesse experimento não validaria a hipótese original de busca em espaço
latente. Seria evidência sobre um mecanismo diferente, voltado ao mesmo problema geral.
Os textos originais estão preservados em [research/](../research/).

## O que significaria melhorar

Seja \(p_\theta(e\mid\mathcal D,G)\) a distribuição de expressões produzidas pelo modelo
sob restrição gramatical, dadas observações \(\mathcal D\). Não buscamos apenas elevar a
probabilidade de uma resposta específica. Queremos testar se a distribuição modificada
produz candidatos de menor erro em novas tarefas, com o mesmo orçamento de geração.

Para cada tarefa, geramos \(K\) candidatos e selecionamos pelas observações disponíveis:

\[
\hat e_\theta = \arg\min_{e\in\{e_1,\ldots,e_K\}}
\left[\operatorname{NMSE}_{\mathrm{obs}}(e)+\lambda C(e)\right].
\]

Depois medimos o erro de \(\hat e_\theta\) em pontos numéricos que não foram usados
nessa seleção. Há dois sentidos distintos de generalização:

1. **Novos pontos da mesma função:** interpolação e extrapolação numéricas.
2. **Novas funções:** transferência da intervenção para tarefas que não orientaram a edição.

Acertar novos pontos de uma função conhecida não demonstra transferência entre funções.
Da mesma forma, trocar uma preferência global de `x` por `sin(x)` pode ajudar tarefas
periódicas e prejudicar todas as outras. Esse efeito deve aparecer nos controles.

## Hipóteses e explicações alternativas

| Hipótese | Evidência necessária | Explicação alternativa a descartar |
| --- | --- | --- |
| A intervenção muda a distribuição de propostas | Mudança repetida de candidatos, logits ou frequências | Flutuação de amostragem |
| A mudança melhora a busca no domínio estudado | Menor erro com K fixo em funções não usadas no desenvolvimento | Mais geração, mudança de prompt ou memorização |
| A escolha mecanística da edição agrega valor | Vantagem sobre controles aleatórios comparáveis | Perturbação genérica dos pesos |
| A melhora é reutilizável | Transferência entre funções e famílias, sem regressões relevantes | Preferência por uma família específica |

A hipótese nula operacional é que a edição não reduz o erro esperado em tarefas
reservadas, sob condições comparáveis. Um resultado negativo pode indicar tanto uma
intervenção ineficaz quanto falta de sensibilidade do piloto. Não prova que WSP seja
impossível em qualquer modelo ou domínio.

## A limitação decisiva da primeira versão

O backend atual gera dentro de um inventário finito de expressões. Cada expressão é
validada e compilada em sequências do tokenizer original. A máscara permite apenas
continuações que terminam em uma expressão desse inventário. Isso trata corretamente
funções e operadores que ocupam vários tokens, mas não oferece síntese estrutural aberta.

Com o inventário padrão de 14 expressões, a pergunta executável é mais modesta:
**a edição melhora a seleção de candidatos dentro de um catálogo fixo?**

Antes de atribuir qualquer resultado ao gerador, precisamos medir a cobertura do catálogo.
Para cada tarefa de desenvolvimento ou validação, um avaliador pode enumerar o inventário
e calcular seu melhor erro possível. Esse oráculo é um diagnóstico do espaço de busca,
não uma condição neural e não entra como feedback do modelo. Se nenhum candidato explica
a função, a edição não pode resolver a tarefa dentro desse espaço.

O relatório deve distinguir tarefas representáveis, tarefas aproximáveis e tarefas fora
do inventário, com critérios numéricos fixados antecipadamente. Construir um catálogo
após olhar respostas finais invalidaria a separação experimental. Cobertura do inventário,
baseline uniforme e análises por família ainda precisam de scripts próprios.

Uma segunda etapa poderá substituir o catálogo por uma gramática composicional limitada
por tamanho/profundidade. Só então será adequado investigar busca por estruturas não
enumeradas previamente. Essa extensão não está implementada.

## Condições experimentais

| Condição | Pesos | Geração e avaliação |
| --- | --- | --- |
| A: baseline | Checkpoint original | K propostas sob a gramática fixada |
| B: feedback em inferência | Mesmo checkpoint original | K propostas sequenciais; perdas anteriores das observações entram no contexto |
| C: WSP | Checkpoint com intervenção escolhida no desenvolvimento/validação | Mesmo regime de A |
| D: edição aleatória | Perturbação do baseline com norma comparável à de C | Mesmo regime de A; repetir com sementes independentes |
| E: no-edit | Mesmo estado de A, recarregado | Repetição para verificar o caminho de execução |

A comparação C versus A mede o efeito da edição. C versus D investiga se a direção
escolhida é melhor que perturbar pesos. B versus A mede o valor de feedback sem edição.
E versus A deve reproduzir os mesmos resultados no backend determinístico; diferenças
em hardware real devem ser investigadas e documentadas.

Igualar K controla a quantidade de candidatos, mas não iguala custo computacional:
B usa contextos progressivamente maiores. Registrar tokens, latência e custo de descobrir
a edição é necessário para uma comparação econômica. O runner atual conta gerações e
tempo total; não mede automaticamente tokens do agente, FLOPs ou custo financeiro.

O controle aleatório atual iguala a norma de Frobenius no tensor alterado. Ele não iguala
posto, esparsidade ou geometria. Para uma conclusão mecanística mais forte, incluir
controles com o mesmo suporte e posto da edição, além de várias sementes. Esses controles
adicionais são propostas para o experimento, não funcionalidades já disponíveis no CLI.

## Organização dos dados e do acesso

Separar funções de desenvolvimento, validação e teste final, além de separar pontos
observados e pontos reservados dentro de cada função. O gerador atual usa templates
estruturais distintos entre splits e fluxos aleatórios derivados do nome do split.
Isso reduz duplicações triviais, mas não garante inequivalência algébrica nem diversidade
suficiente para um benchmark amplo.

O agente usa desenvolvimento para diagnosticar e formular intervenções. Validação serve
para escolher ou rejeitar candidatos dentro de um orçamento declarado. Consultá-la
repetidamente também é uma forma de seleção: registrar quantas propostas foram testadas
e reservar teste final independente. Não interpretar o intervalo da edição vencedora
na validação como se não tivesse havido busca.

O teste final deve ficar sob outra conta de sistema ou em um container/máquina sem
montagem de seus dados nas ferramentas do agente. Uma flag de CLI, um arquivo 0700 sob
o mesmo usuário ou uma instrução no prompt não oferecem esse isolamento. O script de
avaliação final é um exemplo de fluxo; o provisionamento dessa separação é externo.

O runner atual agrega interpolação e extrapolação no campo `heldout`. Relatar esses dois
regimes separadamente exige ampliar o formato ou analisar pontos com regras declaradas
antes da execução. O agente não deve usar erros finais para escolher outra edição.

## Piloto proposto

Começar com um modelo causal pequeno e aberto, como SmolLM2-135M-Instruct, apenas se
houver sinal utilizável no baseline. Fixar a revisão local do modelo e tokenizer,
versões das dependências, dispositivo, inventário e serialização do prompt.

Uma configuração inicial razoável para testar viabilidade seria 30 funções de
desenvolvimento, 30 de validação, K = 16 e cinco sementes de geração. Esses valores
não substituem um dimensionamento amostral posterior. Distribuir tarefas pelas cinco
famílias atuais e verificar cobertura antes de editar. Se o catálogo ou o modelo
produzir um efeito de piso, corrigir o desenho antes de interpretar resultados negativos.

Limitar a busca inicial a dez propostas de edição, com orçamento de tempo e de tokens
do agente registrado. Começar cada proposta a partir do baseline, para evitar confundir
edições isoladas com uma sequência cumulativa. Se futuramente houver composições de
edições, registrar toda a linhagem e testar ablações de cada componente.

Após congelar uma intervenção, avaliar uma única vez em funções finais privadas.
A quantidade de funções finais e o limiar de efeito relevante devem ser definidos a
partir da variabilidade do piloto e dos recursos disponíveis, antes de abrir o teste.

## Um ciclo de intervenção concreto

Considere a observação de que o baseline quase nunca propõe expressões trigonométricas,
mesmo quando os dados têm padrão periódico e essas expressões estão no catálogo.

1. Medir a frequência desse erro em desenvolvimento e verificar se ele depende de
   cobertura do inventário, tokenização, prompt ou orçamento.
2. Comparar contextos periódicos e não periódicos; inspecionar logits e representações
   relevantes. As sondas atuais expõem hidden states e logits do último token. O comando
   `probe` usa somente JSON das observações, enquanto `generate` inclui uma instrução;
   alinhar esses contextos é trabalho pendente antes de atribuir um mecanismo à trajetória.
3. Formular uma hipótese verificável, por exemplo: uma direção de saída de um MLP reduz
   excessivamente a massa de probabilidade em continuações trigonométricas nesse contexto.
   Essa frase é uma hipótese de investigação, não um circuito identificado pelo toolkit.
4. Identificar explicitamente tensor, orientação dos eixos, direção e magnitude. Aplicar
   uma projeção ou atualização de posto um pequena, sem modificar o baseline.
5. Avaliar o candidato com o mesmo inventário, prompt, dados, K e sementes. Comparar
   também funções não periódicas e o controle aleatório correspondente.
6. Registrar a decisão, inclusive quando a frequência de seno aumenta mas o erro piora.
   Voltar ao checkpoint pai quando a intervenção for rejeitada.

Não existe neste projeto um algoritmo que encontre automaticamente a direção correta.
Claude Code ou Codex fornece a investigação e a proposta; o motor fornece operações
executáveis e registros auditáveis.

## Métricas e regra de decisão

A métrica principal proposta é a diferença pareada de NMSE nos pontos reservados da
expressão escolhida pelas observações. Valor negativo favorece a edição. A implementação
usa normalização por `max(var(y), 1e-12)` e complexidade dada pela contagem dos nós
matemáticos do AST. Funções quase constantes podem produzir NMSE muito grande; relatar
MSE como métrica complementar.

Guardar também validade numérica, distribuição das perdas de candidatos, complexidade,
diversidade estrutural, resultados por família, magnitude da edição e custo de busca.
Validade sintática garantida pelo catálogo não garante domínio numérico válido.
Diversidade de AST não prova diversidade funcional.

O comparador atual agrega repetições de cada função e calcula um bootstrap pareado por
função, evitando tratar sementes como tarefas independentes. A regra de triagem exige
pelo menos cinco funções pareadas, nenhum par inválido e limite superior do intervalo
bootstrap de 95% abaixo de zero. Esse mínimo é uma proteção operacional, não uma
justificativa de tamanho de amostra. O CLI `paired` calcula a regra; o registro formal
da decisão é feito pela API `record_decision`, já usada na demonstração.

Para o experimento científico, pré-definir também uma melhora mínima relevante e limites
de regressão por família e de aumento de complexidade. Esses critérios adicionais não
são impostos pelo comparador atual. Em teste final, apresentar efeito, incerteza e falhas,
sem escolher novas intervenções a partir do resultado. Não declarar aquisição de uma
política geral apenas porque um intervalo de validação ficou abaixo de zero.

## Como começar com o toolkit

Primeiro, executar a demonstração de infraestrutura em um diretório novo:

```bash
uv sync --dev
uv run pytest -q
uv run symbolic-wsp demo --output runs/wsp-symbolic-demo
```

Ela usa um gerador sintético e não exige pesos pré-treinados. Seus resultados verificam
os caminhos de execução; não medem a hipótese sobre LLMs.

Para um modelo real, instalar o extra HF, preparar uma cópia local com revisão fixada e
usar os mesmos arquivos de inventário em todas as condições. Neste exemplo, `BASE_ID`
e `CHILD_ID` representam IDs retornados pelos comandos anteriores; `inventory.json` é
uma lista JSON de expressões validáveis, preparada antes da avaliação:

```bash
uv sync --extra hf --dev
uv run symbolic-wsp dataset --split dev --seed 17 --count 30 --output runs/dev.json
uv run symbolic-wsp dataset --split validation --seed 17 --count 30 --output runs/validation.json
uv run symbolic-wsp hf-snapshot --model /path/to/pinned-model
uv run symbolic-wsp run --backend hf --model /path/to/pinned-model --checkpoint BASE_ID --expressions inventory.json --dataset runs/validation.json -k 16 --seeds 11 22 33 44 55 --output runs/baseline.json
uv run symbolic-wsp edit BASE_ID --spec intervention.json --hypothesis 'Hipótese mecanística e previsão observável'
uv run symbolic-wsp run --backend hf --model /path/to/pinned-model --checkpoint CHILD_ID --expressions inventory.json --dataset runs/validation.json -k 16 --seeds 11 22 33 44 55 --condition wsp --output runs/edited.json
uv run symbolic-wsp paired runs/baseline.json runs/edited.json
uv run symbolic-wsp rollback CHILD_ID
```

O arquivo `intervention.json` deve usar dimensões e nomes do checkpoint real. O exemplo
de posto um em `examples/` pertence à cabeça sintética da demonstração e não deve ser
aplicado diretamente ao modelo HF. Realizar o diagnóstico em desenvolvimento antes do
passo de edição e executar também feedback, no-edit e controles aleatórios. Esses comandos
ilustram um fluxo; não significam que um experimento com modelo pré-treinado já foi executado.

## O que precisa constar no relatório

- Identidade do checkpoint, tokenizer, inventário, prompt e ambiente de execução.
- Famílias e quantidade de funções, cobertura do catálogo, splits e sementes públicas.
- Todas as propostas, hipóteses, tensores, normas, custos, decisões e linhagens.
- Comparações pareadas com baseline, feedback e controles aleatórios; regressões por família.
- Métricas numéricas e de complexidade, intervalos, pares inválidos e limitações de cobertura.
- Resultado final independente e distinção entre preferência de saída, melhora da distribuição
  de propostas e evidência de busca reutilizável.

Uma edição rejeitada, com hipótese explícita e avaliação reproduzível, também é um
resultado útil. O objetivo do primeiro experimento é descobrir se há um sinal que
justifique ampliar a gramática e o programa experimental.

## Mapa entre proposta e implementação

| Componente | Estado atual |
| --- | --- |
| Checkpoints identificados, edição, comparação e rollback | Implementado |
| Avaliação segura de expressões e penalidade de complexidade | Implementado |
| Geração HF restrita a catálogo; observações em texto | Implementado; integração testada com modelo local minúsculo |
| Condições A–E, repetições e comparação pareada | Implementado |
| Melhora em pesos de um LLM pré-treinado | Ainda não investigada experimentalmente |
| Auditoria de cobertura, oráculo de catálogo e baseline uniforme | A implementar para o piloto científico |
| Sondas alinhadas ao prompt e às trajetórias completas | A ampliar |
| Controles aleatórios com mesmo posto/suporte | A implementar |
| Relatórios separados de interpolação/extrapolação e por família | A ampliar |
| Teste final realmente isolado e custos completos | Requer infraestrutura e instrumentação externas |
| Gramática composicional e busca estrutural aberta | Etapa posterior |

Para detalhes operacionais, consultar [arquitetura](architecture.md),
[metodologia](methodology.md), [verificação executada](verification.md) e
[configuração dos agentes](agent-setup.md).
