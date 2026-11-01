# Dados oficiais da ANTT (opcional)

Coloque aqui o CSV "Volume de tráfego nas praças de pedágio" (CC BY), baixado em
https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio

Formato esperado: separador `;`, encoding latin-1, colunas `concessionaria`, `mes_ano`,
`sentido`, `praca`, `tipo_cobranca`, `categoria_eixo`, `tipo_de_veiculo`, `volume_total`.

Com o arquivo aqui, `python src/volume_dados.py` calcula o VDM real de cada praça/pórtico da EPR
e imprime junto dos cenários. O CSV não vai para o git (está no .gitignore) por causa do tamanho.
