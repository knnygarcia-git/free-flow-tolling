# Como cuidar melhor dos dados de um pórtico

Aqui eu sigo o dado na ordem em que ele nasce: a foto no pórtico, a leitura da placa, a decisão
de cobrar, a checagem, o armazenamento e o acompanhamento do dia a dia. Em cada etapa digo o que
eu faria e, quando testei, o que aconteceu.

Os números entre parênteses vêm dos experimentos da pasta `src/ml/`, feitos com dados simulados.
Servem para dar uma ideia do tamanho do ganho, não como promessa.

---

## 1. No pórtico

Processe a imagem onde ela nasce. As câmeras de um pórtico enxergam uns 55 GB por minuto.
Mandar isso pela internet não tem como. O computador que fica no próprio pórtico acha a placa,
lê, decide e envia só o resultado: uns 2 KB de registro e 1 MB de provas por carro.

Guarde várias fotos de cada carro, mas jogue fora as ruins antes de enviar. De 3 a 6 fotos por
câmera é o que permite a votação do item 2. Foto com placa cortada, borrada ou estourada de luz
só ocupa espaço.

Acerte os relógios. Câmera, antena e laser são combinados pelo horário. Uma diferença de um
décimo de segundo, a 100 km/h, são quase 3 metros, o suficiente para juntar a tag de um carro
com a foto do carro de trás. Todos os aparelhos precisam sincronizar o relógio pela rede e gravar
a hora no mesmo fuso.

Anote as condições junto com cada passagem: se chovia, se era noite, a velocidade, o estado do
aparelho. Sem isso, o alarme do item 6 não teria como saber se a leitura caiu por causa da chuva
ou de uma lente suja.

## 2. Na leitura da placa

Use o desenho da placa Mercosul (três letras, um número, uma letra, dois números). Um número onde
deveria haver letra é erro na certa: 0 vira O, 8 vira B. Só isso tirou 26% dos erros.

Faça uma votação letra por letra entre todas as fotos, dando mais peso às fotos mais nítidas. Um
erro por acaso raramente se repete em todas elas. Com a votação, os erros caíram 30%.

Separe o erro por acaso do erro que se repete. Placa suja é lida errado do mesmo jeito em todas
as fotos, e nenhuma votação resolve. No teste, 81% dos erros que sobraram eram desse tipo. Esses
casos precisam de uma pessoa.

## 3. Na decisão de cobrar

Não trate a certeza da câmera como se fosse a chance real de acerto. Ela costuma ser otimista:
uma leitura com "95% de certeza" acerta bem menos que isso (dá para ver no gráfico 07). Um
modelo que junta todas as pistas e é ajustado com os acertos reais devolve uma chance em que dá
para confiar.

Escolha o ponto de corte pelo custo, e não porque 90% parece um número bom. Com a chance
ajustada, a conta é simples: vale cobrar sozinho quando a chance de acerto passa de
`1 − custo de uma conferência ÷ custo de uma cobrança errada`. Com R$ 0,60 e R$ 30, isso dá
0,98, e foi bem perto daí que o experimento chegou (0,979).

Use a tag e a placa para vigiar uma à outra. A pista que mais pesou no modelo foi a diferença
entre a placa cadastrada na tag e a placa fotografada. É ela que pega a tag colada em outro
carro, coisa que a regra "leu a tag, cobra" nunca vê.

No teste, as cobranças erradas caíram de 422 para 17. Em troca, as fotos mandadas para uma
pessoa conferir subiram de 1,2% para 4,3% das passagens. Somando tudo, o custo caiu 89%.

## 4. Antes de gravar e antes de cobrar

São checagens simples e automáticas, feitas em cada passagem. Quando uma falha, o motivo fica
anotado, e depois dá para contar quais falham mais.

| Checagem | O que olha | Se falhar |
|---|---|---|
| Campos | se veio tudo, no formato certo, de um pórtico e faixa que existem | descarta e avisa a equipe técnica |
| Formato da placa | se segue o padrão Mercosul ou o antigo (três letras e quatro números) | tenta corrigir; se não der, vai para uma pessoa |
| Cadastro | se a placa existe e se o tipo de veículo cadastrado bate com o que o laser mediu | vai para uma pessoa |
| Tag contra placa | se a placa da tag é a mesma da foto (aceitando uma letra diferente) | investigação de fraude |
| Tag contra laser | se o tipo de veículo da tag é o mesmo que o laser mediu | cobra pelo que foi medido e investiga |
| Repetição | a mesma placa no mesmo pórtico em poucos segundos | junta numa passagem só |
| Tempo impossível | a mesma placa em dois pórticos, numa velocidade acima de 200 km/h | suspeita de placa clonada |
| Horário | passagem com data no futuro, fora de ordem ou com relógio errado | corrige pela hora do servidor e avisa |

## 5. No armazenamento

Guarde o pacote completo de fotos só quando ele pode virar prova. As fotos completas (quase 1 MB)
importam quando a cobrança foi pela placa, quando alguém conferiu ou quando o motorista pode
contestar. Se a tag foi lida e a foto da placa confirmou, com chance de acerto acima de 99,9%,
uma versão enxuta basta: os recortes da placa, o perfil do laser e o registro, uns 35 KB. Nos
4 pórticos simulados, isso baixa de 33 para 14 GB por dia, 59% a menos. Antes de fazer isso de
verdade, é preciso confirmar com o jurídico e com o contrato de concessão o que tem de ser
guardado e por quanto tempo.

Organize pensando em quem vai consultar. Os registros podem ficar em arquivos Parquet separados
por data e por pórtico. Assim, uma pergunta como "quanto a faixa 2 leu ontem" abre só o pedaço
necessário. As fotos ficam num armazenamento de arquivos, com um nome que diz pórtico, data, hora
e passagem, e a tabela guarda só esse nome.

Separe por idade. As fotos recentes, dentro do prazo de contestação, ficam num lugar rápido.
Depois vão para um armazenamento mais barato e lento. No fim do prazo, são apagadas.

Trate a placa como dado pessoal. Placa com horário e local conta a rotina de alguém, e a LGPD se
aplica. Limite quem pode ver, registre quem consultou e, nas bases usadas para análise e para
treinar modelos, troque a placa por um código que não permita voltar a ela.

## 6. No dia a dia

Compare cada câmera com o que ela deveria estar lendo, e não com um número fixo. Um alarme do
tipo "abaixo de 85%" toca toda vez que chove: foram 172 alarmes falsos em 3 dias. Quando
comparei com o esperado para aquela hora e aquele tempo, os alarmes falsos sumiram, e as duas
câmeras com defeito foram percebidas em 10 e em 30 minutos.

Estes são os números que eu colocaria num painel, por pórtico e por faixa:

| Número | O que ele conta |
|---|---|
| Quanto foi cobrado sem ninguém conferir | se o processo está andando sozinho |
| Quanto foi para uma pessoa, e o tamanho da fila | custo e atraso da cobrança |
| Cobranças erradas a cada 10 mil (contestações que o motorista ganhou) | quanto o sistema acerta de verdade |
| Leituras seguras de cada câmera, contra o esperado | se a câmera ou o flash estão bons |
| Quanto a foto da frente concorda com a de trás | qualidade da leitura de placa |
| Quantas tags foram lidas, entre os carros que têm tag | se a antena está boa |
| Tag que não bate com a placa ou com o laser | fraude |
| Tempo entre a passagem e o registro chegar ao sistema | se a rede e o pórtico estão bem |

## 7. Com o tempo

Cada foto conferida por uma pessoa é uma resposta certa que dá para usar no treino. Vale mandar
primeiro para conferência os casos em que o modelo está mais em dúvida. São os que mais ensinam.

Só que tem uma armadilha: só se sabe a verdade dos casos que alguém conferiu ou que o motorista
contestou. Para não enganar o modelo, alguém precisa conferir também uma amostra aleatória das
cobranças feitas automaticamente. É assim que se mede o erro que ninguém reclamou.

E o mundo muda. Câmeras envelhecem, a frota troca, aparecem placas novas. Eu olharia todo mês se a
chance prevista pelo modelo continua batendo com os acertos reais (o gráfico 07) e treinaria de
novo quando ela começasse a se afastar.
