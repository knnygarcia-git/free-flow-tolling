# Como o pórtico descobre quem passou e quanto cobrar

O pórtico é aquela estrutura de metal que atravessa a pista. Pendurados nele, sobre cada faixa,
ficam alguns aparelhos. Cada um responde a uma pergunta diferente: quem é esse carro (tag ou
placa), que tipo de veículo ele é (quantos eixos) e como provar que ele passou (fotos e horário).
Tudo isso com o carro a 80 ou 120 km/h, sem cancela e sem frear.

```
             ┌──────────────────────── PÓRTICO ─────────────────────────┐
             │  [antena]   [câmera da placa + flash IR]   [laser]   [câmera geral]  │   um conjunto por faixa
             └───────────────────────────────────────────────────────────┘
                    │                 │                 │
  0 a 16 m antes ───┼── área da antena┼─ foto da placa ─┼──── embaixo do pórtico ──→ sentido do tráfego
                    │                 │  (10 m antes)   │
                    ▼                 ▼                 ▼
              lê a tag          fotografa a placa   mede o veículo e conta os eixos
                    └────────┬────────┴────────┬────────┘
                             ▼                 ▼
                    computador no próprio pórtico junta as três leituras
                             ▼
                    provas (~1 MB) + registro de cobrança (~2 KB) → sistema da concessionária
                             ▼
                 com tag: a cobrança vai para a empresa da tag (Sem Parar, ConectCar, Veloe...)
                 sem tag: cobrança pela placa, 30 dias para pagar; quem não paga leva multa (art. 209-A do CTB)
```

## A tag

A tag é aquele adesivo no para-brisa. Dentro dele há um chip e uma antena impressa, e nada de
bateria.

A antena do pórtico emite ondas de rádio (na faixa de 902 a 928 MHz, a mesma das etiquetas
antifurto de loja). A própria onda dá energia ao chip, que "acorda" e responde refletindo o sinal
com o seu código. O padrão técnico tem nome, ISO/IEC 18000-63, e é o mesmo usado em logística.

A leitura leva milésimos de segundo e se repete várias vezes enquanto o carro atravessa a área
da antena. O sistema também anota a força do sinal, que ajuda a saber em qual faixa o carro
estava.

Qualquer tag autorizada funciona em qualquer pórtico de rodovia federal, por causa de uma regra
da ANTT de 2014 (Resolução 4.281). Quem usa tag também ganha desconto.

## A placa

Ler placa por câmera tem nome técnico, ANPR, e é o mesmo tipo de problema do vídeo dos peixes:
achar o objeto na imagem, acompanhar enquanto ele se move e extrair uma informação dele.

1. A foto. As câmeras são em preto e branco e têm um flash infravermelho que pisca junto com o
   disparo. A exposição dura milionésimos de segundo, o suficiente para congelar um carro a
   120 km/h. O infravermelho funciona de dia e de noite, não se atrapalha com farol nem com sol
   de frente e faz a película da placa brilhar. Em geral há uma câmera para a placa da frente e
   outra para a de trás, em cada faixa.
2. O momento certo. O próprio vídeo, um laser ou um radar percebe o carro chegando e escolhe as
   melhores fotos quando ele cruza uma linha imaginária, uns 10 metros antes do pórtico.
3. Achar a placa. Um programa de visão computacional (do tipo YOLO) desenha um retângulo em volta
   da placa.
4. Endireitar. O recorte é corrigido para não ficar torto nem escuro.
5. Ler. Outro programa lê as sete letras e números. O formato Mercosul ajuda a conferir: em
   `ABC1D23`, a terceira posição é sempre letra, então um zero ali só pode ser a letra O.
6. Decidir. Cada leitura vem com um grau de certeza. As leituras das várias fotos, da frente e
   de trás, são combinadas. Se a certeza ficar baixa, uma pessoa olha a foto.

As trocas mais comuns são O com 0, B com 8, I com 1, S com 5 e Z com 2. Placa suja, amassada,
tampada ou escondida por reboque também atrapalha. Quando o carro tem tag, a placa cadastrada
junto com a tag serve de conferência. Se a tag diz `ABC1D23` e a câmera leu `ABC1023`, vale a
tag. Se as duas não têm nada a ver uma com a outra, pode ser tag colada no carro errado ou placa
clonada, e o caso vai para investigação.

## O tipo de veículo

Um laser sobre cada faixa varre o veículo e monta um desenho em 3D dele: comprimento, largura,
altura e quantos eixos tem. Com isso o sistema sabe a categoria da tarifa. Carro de passeio paga
uma coisa, caminhão e ônibus pagam por eixo, e eixo levantado não conta.

A categoria cadastrada na tag é comparada com a que o laser mediu. É assim que se pega, por
exemplo, um caminhão usando a tag de um carro.

## A decisão

O computador do pórtico junta as três leituras pelo horário e pela faixa:

| O que aconteceu | O que o sistema faz |
|---|---|
| Leu a tag e ela é válida | cobra pela tag e usa a placa para conferir |
| Sem tag, placa lida com segurança | cobra pela placa |
| Sem tag, leitura duvidosa | uma pessoa confere as fotos |
| Tag e placa não batem | investigação (pode ser fraude) |

O resultado de cada passagem é um registro pequeno, de uns 2 KB, com pórtico, faixa, data, hora,
placa ou tag, categoria e grau de certeza. Junto vai um pacote de provas de cerca de 1 MB, com as
fotos, que fica guardado caso alguém conteste a cobrança. A conta de quanto isso dá por minuto
está em [`src/volume_dados.py`](../src/volume_dados.py) e no README.

## Onde entra a ciência de dados

São milhões de passagens por mês saindo dos pórticos para a nuvem. Alguém precisa juntar as
leituras repetidas do mesmo carro e organizar tudo por pórtico e por data.

Dá para medir quanto a câmera acerta, quais letras ela mais confunde e quanto a chuva e a noite
pioram a leitura.

Dá para prever quem não vai pagar no prazo e cobrar antes, achar fraude (tag, placa e categoria
que não batem; a mesma placa em dois pórticos distantes num intervalo impossível) e prever o
movimento de cada pórtico por hora e por época do ano.

Três dessas ideias estão testadas na pasta [`src/ml/`](../src/ml/) e explicadas no README.
