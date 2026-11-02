# Como otimizar os dados de um pórtico Free Flow

Este documento junta o que os experimentos do projeto mostraram com práticas de engenharia de
dados para sistemas de leitura de placa e tag. A ideia é seguir o dado na ordem em que ele
nasce: captura, leitura, decisão, validação, armazenamento e operação.

Os números entre parênteses vêm dos experimentos em `src/ml/`, rodados sobre dados
sintéticos. Servem como ordem de grandeza.

---

## 1. Na captura (computador de borda, no pórtico)

**Processe onde o dado nasce.** As câmeras de um pórtico de um sentido produzem ~55 GB de imagem
bruta por minuto. Mandar isso para a nuvem é inviável. O computador de borda detecta, lê e
envia só o resultado: ~2 KB de transação e ~1 MB de evidência por veículo.

**Guarde vários quadros, mas escolha bem.** Ter de 3 a 6 leituras por câmera é o que permite a
votação do experimento 2. Descarte, ainda na borda, os quadros com placa cortada, borrada ou
superexposta, para não gastar banda com imagem que não ajuda.

**Sincronize os relógios.** Câmera, antena e laser são combinados pelo horário e pela faixa.
Uma diferença de 100 ms equivale a ~3 m de deslocamento a 100 km/h e basta para associar a
tag ao carro errado. Use NTP/PTP em todos os equipamentos e registre o horário em UTC.

**Registre o contexto.** Chuva, luminosidade, velocidade e o estado do equipamento entram no
evento como colunas. Sem eles, o experimento 3 não teria como distinguir chuva de lente suja.

## 2. Na leitura da placa

| Prática | Por quê | Resultado no projeto |
|---|---|---|
| Regra de formato Mercosul (LLLNLNN) | Dígito em posição de letra é erro certo (0→O, 8→B) | −26% de placas erradas |
| Votação por caractere entre quadros e câmeras, ponderada pela confiança | Erros aleatórios não se repetem em todos os quadros | −30% de placas erradas |
| Separar erro aleatório de erro persistente | Placa suja é lida errada sempre do mesmo jeito; mais quadros não resolvem | 81% dos erros restantes estão em placas ruins |

## 3. Na decisão de cobrar

**Não use a confiança do OCR como probabilidade.** Ela costuma ser superconfiante: uma leitura
com "95%" pode acertar bem menos que isso (figura 07). Calibre um modelo que combine todos os
sinais e devolva uma probabilidade em que se possa confiar.

**Escolha o limiar pelo custo, não por costume.** Com a probabilidade calibrada, a regra ótima
é aceitar quando `P(acerto) ≥ 1 − custo_revisão / custo_erro`. Com R$ 0,60 por revisão e
R$ 30 por cobrança errada, isso dá 0,98, e foi o que o experimento 1 encontrou (0,979).

**Use a tag e a placa uma para conferir a outra.** A variável mais importante do modelo foi a
distância entre a placa cadastrada na tag e a lida pelo OCR. É ela que pega tag instalada em
outro veículo, um erro que a regra "tag manda" nunca vê.

Resultado: cobranças erradas de 422 para 17 no dia de teste, com a revisão humana subindo de
1,2% para 4,3% das passagens. O custo total cai 89%.

## 4. Na validação (antes de gravar e antes de cobrar)

Regras simples, automáticas, aplicadas a cada evento. Cada falha vira um motivo registrado,
o que permite contar e priorizar.

| Regra | Verifica | Ação se falhar |
|---|---|---|
| Esquema | campos obrigatórios, tipos, pórtico e faixa existentes | rejeita e alerta a equipe técnica |
| Formato | placa no padrão Mercosul ou no antigo (LLLNNNN) | tenta corrigir; senão, revisão |
| Cadastro | a placa existe? a categoria cadastrada bate com a do laser? | revisão |
| Tag × placa | placa da tag igual à lida pelo OCR (tolerando 1 caractere) | auditoria de fraude |
| Tag × laser | categoria da tag igual à medida | cobra pela medida e auditoria |
| Duplicidade | mesma placa no mesmo pórtico em menos de N segundos | une em uma passagem só |
| Tempo impossível | mesma placa em dois pórticos a uma velocidade implícita acima de 200 km/h | suspeita de placa clonada |
| Horário | evento no futuro, fora de ordem ou com relógio dessincronizado | corrige pelo servidor e alerta |

## 5. No armazenamento

**Guarde a prova completa só quando ela pode ser necessária.** O pacote completo (~965 KB, com
foto colorida e fotos ANPR) importa em cobrança pela placa, revisão e contestação. Para tag
confirmada pela placa com probabilidade ≥ 0,999, uma versão reduzida (~35 KB: recortes da
placa, perfil do laser e metadados) basta. No projeto, isso reduz o volume diário em **59%**
(33 → 14 GB/dia em 4 pórticos). Antes de adotar, confirme com o jurídico e com o contrato de
concessão o que precisa ser guardado e por quanto tempo.

**Organize para a consulta.**
- Transações em formato colunar (Parquet), particionadas por `data/portico`. Consultas do tipo
  "taxa de leitura da faixa 2 ontem" leem só a partição necessária.
- Imagens em armazenamento de objetos, com a chave `portico/data/hora/id_passagem`, e só a
  referência na tabela.
- Camadas por idade: imagens recentes em armazenamento rápido (período de contestação),
  depois em camada fria e mais barata, e por fim exclusão.

**Trate a placa como dado pessoal (LGPD).** Placa ligada a horário e local revela a rotina de
uma pessoa. Restrinja o acesso, registre quem consultou e pseudonimize (por exemplo, com hash
com chave secreta) nas bases usadas para análise e treino de modelos.

## 6. Na operação

**Monitore cada sensor contra o esperado, não contra um número fixo.** Um limiar fixo ("taxa <
85%") dispara com toda chuva: foram 172 alarmes falsos em 3 dias. Comparar com a taxa esperada
para as condições do momento, com CUSUM, zerou os alarmes falsos e detectou as duas falhas em
10 a 30 minutos (experimento 3).

**Indicadores para um painel diário, por pórtico e faixa:**

| Indicador | O que revela |
|---|---|
| Cobrança automática (%) | eficiência do processo |
| Revisão humana (%) e tamanho da fila | custo e atraso da cobrança |
| Cobranças erradas por 10 mil (contestações procedentes) | acurácia real do sistema |
| Leituras confiáveis por câmera (esperado × observado) | saúde da câmera e da iluminação |
| Concordância frente × traseira | qualidade do OCR |
| Leitura de tag (% dos veículos com tag) | saúde da antena |
| Divergência tag × placa e tag × laser | fraude |
| Atraso entre a passagem e o evento chegar ao backoffice | saúde da rede e da borda |

## 7. Na melhoria contínua

**A revisão humana é a fonte de rótulos.** Cada imagem revisada vira um exemplo com a placa
correta para retreinar o OCR e o modelo de fusão. Mande para revisão primeiro os casos em que o
modelo está mais incerto: são os que mais ensinam (aprendizado ativo).

**Cuidado com o viés do rótulo.** Só se conhece a verdade dos casos revisados ou contestados. Uma
amostra aleatória de passagens aceitas automaticamente também deve ser revisada, para medir o
erro que ninguém reclamou.

**Recalibre com o tempo.** Câmeras envelhecem, a frota muda e entram placas novas. Acompanhe a
calibração (figura 07) mês a mês e retreine quando ela se afastar da diagonal.
