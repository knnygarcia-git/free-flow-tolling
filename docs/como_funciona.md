# Como um pórtico Free Flow identifica e cobra um veículo

O pórtico é uma estrutura metálica sobre a pista. Nela ficam três famílias de sensores, cada uma
com uma tarefa: **identificar** o veículo (tag ou placa), **classificar** (quantos eixos, qual
categoria) e **registrar a evidência** (fotos, horário, posição). Tudo acontece com o carro a
80–120 km/h, sem cancela nem redução de velocidade.

```
             ┌──────────────────────── PÓRTICO ────────────────────────┐
             │  [antena RFID]  [câmera ANPR + IR]  [laser 3D]  [contexto]│   ← um conjunto por faixa
             └──────────────────────────────────────────────────────────┘
                    │                 │                 │
  0–16 m antes ────┼── zona RFID ────┼─ disparo ANPR ──┼──── sob o pórtico ──→ sentido do tráfego
                    │                 │  (~10 m antes)  │
                    ▼                 ▼                 ▼
              lê a tag (ms)     foto da placa     perfil, eixos, altura
                    └────────┬────────┴────────┬────────┘
                             ▼                 ▼
                    COMPUTADOR DE BORDA (no pórtico): fusão tag × placa × categoria
                             ▼
                    pacote de evidência (~1 MB) + transação (~2 KB) → backoffice da concessionária
                             ▼
                 com tag: cobra a operadora (Sem Parar, ConectCar, Veloe...)
                 sem tag: cobrança pela placa, prazo típico de 30 dias; não pagar = infração (art. 209-A do CTB)
```

## 1. Leitura de tag (RFID UHF)

* A tag é um adesivo no para-brisa com um chip e uma antena impressa. **Não tem bateria.**
* A antena do pórtico emite ondas de rádio na faixa de **UHF (902–928 MHz no Brasil)**. A tag
  usa a própria energia dessa onda para "acordar" e responde modulando o sinal refletido
  (*backscatter*), no protocolo **ISO/IEC 18000-63 (EPC Gen2)**, o mesmo das etiquetas de logística.
* A resposta traz o identificador único do chip (TID/EPC). A leitura leva milissegundos e é
  repetida várias vezes enquanto o carro cruza a zona da antena, que tem alguns metros de comprimento.
  O sistema guarda também a força do sinal (RSSI), usada para saber **em qual faixa** a tag estava.
* A interoperabilidade (qualquer tag autorizada funciona em qualquer pórtico federal) vem da
  Resolução ANTT nº 4.281/2014. Com tag, o motorista recebe descontos (DBT/DUF) e a cobrança vai
  para a fatura.

## 2. Leitura de placa (ANPR/OCR)

ANPR (*Automatic Number Plate Recognition*) é a mesma tarefa de visão computacional do vídeo dos
peixes, com etapas parecidas: detectar, rastrear, extrair a informação.

1. **Captura.** Câmeras monocromáticas com obturador global (*global shutter*) e iluminador
   **infravermelho** pulsado em sincronia com a exposição. A exposição é curtíssima (dezenas a
   centenas de microssegundos), o que congela um carro a 120 km/h. O infravermelho funciona de
   dia e de noite, ignora farol e sol de frente e destaca a película retrorrefletiva da placa.
   Normalmente há uma câmera para a placa **dianteira** e outra para a **traseira** em cada faixa.
2. **Disparo.** Um laser, radar ou o próprio vídeo detecta o veículo e escolhe os melhores quadros
   quando ele cruza a linha de disparo.
3. **Detecção da placa.** Uma rede neural (do tipo YOLO/SSD) localiza o retângulo da placa no quadro.
4. **Retificação.** O recorte é corrigido em perspectiva e contraste.
5. **OCR.** Outra rede (CRNN ou transformer) lê os 7 caracteres. O formato Mercosul (`LLLNLNN`,
   ex.: `ABC1D23`) serve de regra de validação: um `0` na terceira posição tem de ser `O`.
6. **Confiança e votação.** Cada leitura tem uma confiança. Leituras de vários quadros e das duas
   câmeras (frente e traseira) são combinadas. Abaixo de um limiar, a imagem vai para
   **revisão humana**.

Erros clássicos de OCR são as confusões `O↔0`, `B↔8`, `I↔1`, `S↔5` e `Z↔2`, além de placa suja, amassada, tampada ou com
reboque. Quando o veículo tem tag, a placa vinculada à tag no cadastro da operadora serve de
**conferência**: se a tag diz `ABC1D23` e o OCR leu `ABC1023`, prevalece a tag; se as duas
divergem muito, pode ser tag transferida de carro ou placa clonada, e o caso vai para auditoria.

## 3. Classificação (categoria da tarifa)

* **Scanners laser (LiDAR)** montados sobre cada faixa varrem o veículo e montam um perfil 3D:
  comprimento, largura, altura e número de eixos.
* Com isso o sistema define a categoria ANTT (ex.: Cat 1 = automóvel de 2 eixos; caminhões e
  ônibus pagam por eixo; eixo suspenso não é cobrado).
* A categoria declarada no cadastro da tag é comparada com a medida, o que pega, por exemplo,
  um caminhão usando a tag de um carro.

## 4. Fusão e evento

O computador de borda junta as três leituras pelo **horário e pela faixa** e decide:

| Situação | Decisão |
|---|---|
| Tag lida e válida | cobra pela tag; placa = conferência |
| Sem tag, OCR com confiança ≥ limiar | cobra pela placa |
| Sem tag, OCR com confiança baixa | revisão manual das imagens |
| Tag e placa divergentes | auditoria (fraude, tag em outro veículo) |

O resultado é um **evento de passagem**: um registro pequeno (~2 KB) com pórtico, faixa, data e
hora, identificação, categoria e confiança, mais um **pacote de evidência** (~1 MB) com as fotos,
que fica guardado para contestações. Quanto isso dá por minuto está em
[`src/volume_dados.py`](../src/volume_dados.py) e no README.

## 5. Onde entra a ciência de dados

* **Engenharia de dados:** milhões de eventos por mês, streaming da borda para a nuvem, deduplicação
  (mesmo carro lido por duas câmeras), particionamento por pórtico e data.
* **Qualidade de OCR:** taxa de leitura, matriz de confusão por caractere, impacto de chuva e da noite.
* **Evasão e inadimplência:** modelos para prever quem não vai pagar no prazo e priorizar a cobrança.
* **Fraude:** divergência tag × placa × categoria; placas clonadas (mesma placa em dois pórticos
  distantes em tempo impossível).
* **Tráfego:** curvas horárias, sazonalidade e previsão de demanda por pórtico.
