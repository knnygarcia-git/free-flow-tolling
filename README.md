# Free Flow: quanto dado um pedágio sem cancela gera, e como ele "enxerga" os veículos

Projeto de análise de dados e visão computacional sobre o **pedágio Free Flow** no Brasil, com
foco nos pórticos da **EPR Paraná** (Lote 4: BR-369 e BR-376), que começaram a cobrar em 4 de maio de 2026.

![Simulação do pórtico](reports/figures/simulacao_portico.gif)

*Simulação: o pórtico aparece na cena com os sensores de cada faixa (① câmera ANPR com flash
infravermelho, ② antena RFID, ③ laser 3D, ④ câmera de contexto) e a área que cada um cobre: a
antena lê de 0 a 16 m antes do pórtico, a câmera fotografa a placa a 10 m e o laser mede os eixos
exatamente sob a viga. Uma vista lateral embaixo mostra as mesmas distâncias. Também há
detecção e rastreamento de veículos, OCR da placa, leitura de tag RFID,
classificação por eixos e geração de dados em tempo real. **A cena é sintética**; a lógica do
pipeline segue a de um pórtico real (ver [`docs/como_funciona.md`](docs/como_funciona.md)).*

## Perguntas

1. Onde estão e quantos são os pórticos Free Flow no Brasil e no Paraná?
2. **Quantos dados um pórtico recebe por minuto?** Do sensor bruto ao registro de cobrança.
3. Como os sensores leem placas e tags a 100 km/h?
4. Como machine learning melhora a acurácia e reduz o custo e o volume de dados?

## Principais resultados

**Cadastro (dados reais, base aberta CC BY 4.0, atualizada em 26/08/2026)**

* 69 locais e **90 pórticos** cadastrados em 8 UFs, de 17 concessionárias; 61 locais já cobram.
* **Paraná: 14 pórticos em 10 locais**, sendo 12 do grupo EPR (8 da EPR Paraná e 4 da EPR Iguaçu).
* A EPR Paraná usa **um pórtico por sentido** em cada um dos 4 locais do Lote 4.

| | |
|---|---|
| ![](reports/figures/01_porticos_por_uf.png) | ![](reports/figures/03_porticos_parana.png) |

**Volume de dados (modelo paramétrico, cenários de tráfego)**

Cada passagem gera um **pacote de evidência de cerca de 965 KB** (2 fotos ANPR, recortes da placa, foto de
contexto, perfil do laser e metadados) e uma **transação de cerca de 2 KB** para o backoffice.

| Cenário (VDM por local, 2 sentidos) | Veículos/min no pico | Evidências no pico | Evidências por dia | Rede EPR Lote 4 por ano |
|---|---:|---:|---:|---:|
| Baixo: 8.000 | 10 | ~9,5 MB/min | ~7,7 GB | ~11 TB |
| Médio: 20.000 | 25 | ~24 MB/min | ~19 GB | ~28 TB |
| Alto: 40.000 | 49 | ~48 MB/min | ~39 GB | ~56 TB |

O dado útil é uma fração mínima do que os sensores capturam. Num pórtico de um sentido com 5
câmeras, o fluxo **bruto** é de cerca de **55 GB por minuto**. A borda (computador no pórtico) reduz isso
para algumas dezenas de MB de evidências e algumas dezenas de KB de transações:

| | |
|---|---|
| ![](reports/figures/05_camadas_de_dados.png) | ![](reports/figures/04_dados_por_minuto_24h.png) |

> **Sobre os números de tráfego:** a contagem de veículos por pórtico da EPR não estava acessível
> neste ambiente, por isso os VDMs acima são **hipóteses declaradas** (rodovia regional, BR-369
> e eixo Maringá–Apucarana). Para usar o dado oficial, baixe o CSV da ANTT em
> `data/raw/antt/` (instruções em [`data/raw/antt/LEIA-ME.md`](data/raw/antt/LEIA-ME.md)) e
> rode `src/volume_dados.py` de novo. Os tamanhos de arquivo são valores típicos de sistemas
> ANPR, e todos são parâmetros editáveis no topo do script.

## Como os sensores funcionam (resumo)

| Sensor | O que faz | Tecnologia |
|---|---|---|
| Antena RFID | Lê a tag do para-brisa | UHF 902–928 MHz, ISO/IEC 18000-63; tag passiva, energizada pela onda da antena |
| Câmeras ANPR | Fotografam placa dianteira e traseira | Monocromáticas, *global shutter*, flash infravermelho, exposição de microssegundos |
| OCR | Lê os 7 caracteres da placa | Rede de detecção (YOLO) + rede de leitura (CRNN), validação pelo formato Mercosul |
| Laser 3D | Mede perfil e eixos para definir a categoria | LiDAR sobre cada faixa |
| Computador de borda | Junta tudo e gera o evento | Fusão por horário e faixa; tag manda, placa confere |

Explicação completa, com diagrama e tabela de decisão: **[`docs/como_funciona.md`](docs/como_funciona.md)**.

## Machine learning: mais acurácia, menos custo

Três experimentos sobre uma base sintética de **104 mil passagens** (3 dias, 4 pórticos, com
noite, chuva, placas sujas, tags em outro veículo e duas falhas de sensor no último dia). O
gerador conhece a placa verdadeira; na operação real esse rótulo vem da revisão humana e das
contestações. Todos usam **validação temporal**: treinam nos dias 1 e 2 e testam no dia 3.

> Os números abaixo medem o método sobre dados simulados. Eles mostram a ordem de grandeza do
> ganho, não o desempenho de um pórtico real. Os custos (R$ 0,60 por revisão, R$ 30 por cobrança
> errada) são hipóteses editáveis.

### 1. Fusão das leituras com limiar por custo ([`exp1_fusao_limiar.py`](src/ml/exp1_fusao_limiar.py))

Um gradient boosting calibrado estima a chance de a placa cobrada estar certa, usando 18 sinais:
confiança das duas câmeras, concordância entre quadros, distância entre a placa da tag e a lida
pelo OCR, se a placa existe no cadastro, se a categoria do cadastro bate com a do laser, chuva,
noite etc. O limiar de aceitação automática é o que minimiza o custo total.

| Dia de teste (34.644 passagens) | Cobrança automática | Revisão humana | Cobranças erradas | Custo por 10 mil |
|---|---:|---:|---:|---:|
| Regra atual (tag ou OCR ≥ 90%) | 98,8% | 1,2% | 422 | R$ 3.729 |
| Regra com o melhor limiar de OCR | 92,3% | 7,7% | 188 | R$ 2.091 |
| **Modelo, limiar ótimo (0,979)** | **95,7%** | **4,3%** | **17** | **R$ 405** |

O modelo corta **96% das cobranças erradas** e **89% do custo**. Dos erros que a regra deixava
passar, 323 eram OCR errado com confiança alta e 85 eram tag instalada em outro veículo. A
confiança crua do OCR é superconfiante; a probabilidade do modelo é calibrada (figura 07).

| | |
|---|---|
| ![](reports/figures/06_custo_por_limiar.png) | ![](reports/figures/08_importancia_variaveis.png) |

### 2. Correção de caracteres pelo formato Mercosul ([`exp2_correcao_mercosul.py`](src/ml/exp2_correcao_mercosul.py))

Só passagens sem tag (13.182 no dia de teste), que dependem 100% do OCR.

| Método | Acurácia da placa | Erros por 10 mil | Redução |
|---|---:|---:|---:|
| A. Melhor quadro | 95,4% | 456 | – |
| B. + regra Mercosul (dígito em posição de letra vira letra) | 96,6% | 335 | −26% |
| **C. Votação ponderada entre quadros + regra** | **96,8%** | **321** | **−30%** |
| D. Modelo de confusão aprendido | 96,7% | 329 | −28% |

A votação com a regra resolve os erros aleatórios. **81% dos erros que sobram estão em placas
sujas, danificadas ou tampadas**, que são lidas erradas do mesmo jeito em todos os quadros.
Esses casos não se resolvem com mais quadros: vão para revisão humana, e é o modelo do
experimento 1 que os separa.

| | |
|---|---|
| ![](reports/figures/09_correcao_mercosul.png) | ![](reports/figures/10_confusoes_ocr.png) |

### 3. Alerta de sensor degradado ([`exp3_saude_sensores.py`](src/ml/exp3_saude_sensores.py))

O indicador é a fração de leituras confiáveis por faixa, a cada 10 minutos. Ele também cai com
chuva e à noite, então um modelo aprende a taxa esperada para as condições do momento e um
CUSUM acumula os desvios.

| | Alarme ingênuo (taxa < 85%) | Modelo + CUSUM |
|---|---:|---:|
| Janelas com alarme falso (3 dias, 8 faixas) | 172 | **0** |
| Iluminador IR queimado (abrupto, à noite) | detecta, mas já disparava antes | **1ª janela de 10 min** |
| Lente suja (gradual) | 20 min, misturado a alarmes falsos | **30 min** |

O alarme ingênuo é mais rápido na lente suja só porque dispara o tempo todo. Com 172 alarmes
falsos, ninguém confia nele.

![](reports/figures/11_saude_sensores.png)

## Como otimizar os dados

Resumo das práticas, do sensor ao armazenamento. Detalhes, regras de validação e indicadores em
**[`docs/otimizacao_dados.md`](docs/otimizacao_dados.md)**.

| Etapa | O que fazer | Ganho medido ou esperado |
|---|---|---|
| Captura | Processar na borda e guardar só o evento: de ~55 GB/min de imagem bruta para ~2 KB por passagem | ordens de grandeza menos tráfego de rede |
| Leitura | Votação entre quadros + regra Mercosul | −30% de placas erradas (exp. 2) |
| Decisão | Probabilidade calibrada + limiar por custo, em vez de "confiança ≥ 90%" | −96% de cobranças erradas, −89% de custo (exp. 1) |
| Validação | Regras automáticas: formato, cadastro, categoria tag × laser, duplicidade, horário | pega fraude de tag e leitura impossível antes da cobrança |
| Armazenamento | Pacote completo de fotos só quando a prova pode ser necessária; versão reduzida para tag confirmada | −59% de GB por dia |
| Operação | Monitorar a saúde de cada câmera com esperado × observado | falha detectada em 10–30 min, sem alarmes falsos (exp. 3) |
| Melhoria contínua | Usar a revisão humana como rótulo e revisar primeiro os casos incertos | o modelo melhora com o uso |

![](reports/figures/12_politica_armazenamento.png)

## Estrutura

```
data/raw/sem_parar/      cadastro aberto de pórticos, rodovias e concessionárias (CC BY 4.0)
data/raw/antt/           (opcional) CSV oficial de volume de tráfego da ANTT
src/analise_porticos.py  análise exploratória do cadastro -> figuras 01-03
src/volume_dados.py      modelo de volume de dados em 4 camadas -> figuras 04-05 e tabela
src/simulador_portico.py simulação visual do pórtico -> MP4, GIF, PNG e eventos_simulados.csv
reports/figures/         gráficos e vídeo
reports/tabelas/         tabelas geradas
src/ml/gerar_eventos.py  base sintética de passagens para ML (data/sintetico/, fora do git)
src/ml/exp1_*.py         fusão das leituras + limiar por custo + política de armazenamento
src/ml/exp2_*.py         correção de caracteres Mercosul
src/ml/exp3_*.py         saúde dos sensores (esperado × observado + CUSUM)
docs/como_funciona.md    como tag, placa e laser são lidos
docs/otimizacao_dados.md como otimizar os dados, do sensor ao armazenamento
```

## Como rodar

```bash
pip install -r requirements.txt
python src/analise_porticos.py
python src/volume_dados.py
python src/simulador_portico.py --segundos 30 --seed 42

# machine learning (cerca de 1 minuto no total)
python src/ml/gerar_eventos.py
python src/ml/exp1_fusao_limiar.py
python src/ml/exp2_correcao_mercosul.py
python src/ml/exp3_saude_sensores.py
```

## Limitações e próximos passos

* Os VDMs são cenários; o próximo passo é calibrar com o CSV diário da ANTT.
* A simulação usa uma cena gerada por código; um passo seguinte seria trocar por vídeo real de
  rodovia com um detector treinado (YOLOv8) e um OCR de placas (ex.: dataset brasileiro RodoSol-ALPR).
* Os experimentos de ML usam dados sintéticos. Com dados reais, o rótulo viria da revisão humana,
  e seria preciso cuidar do viés de só conhecer a verdade dos casos que foram revisados.
* Encadear os experimentos: usar a placa corrigida do experimento 2 como entrada do experimento 1.
* Outras ideias: previsão de demanda por pórtico, probabilidade de inadimplência (sem tag),
  detecção de placa clonada (mesma placa em pórticos distantes em tempo impossível).

## Fontes

* Cadastro de pórticos e concessionárias: [Sem-Parar/SEM-PARAR-FREE-FLOW](https://github.com/Sem-Parar/SEM-PARAR-FREE-FLOW) (CC BY 4.0), que cita a [ANTT, Free Flow](https://www.gov.br/antt/pt-br/free-flow)
* Volume de tráfego (para calibrar): [ANTT, dados abertos](https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio)
* Início da cobrança na EPR Paraná: [Bem Paraná](https://www.bemparana.com.br/noticias/parana/free-flow-comeca-a-ser-cobrado-nesta-segunda-feira-no-parana-veja-em-quais-pontos/), [Banda B](https://www.bandab.com.br/noticias-parana/epr-parana-recebe-autorizacao-da-antt-para-cobrar-pedagio-eletronico-em-quatro-porticos/amp/)
