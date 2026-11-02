# Official ANTT traffic data (optional)

Put the CSV "Volume de tráfego nas praças de pedágio" (traffic volume at toll sites, CC BY) here.
Download it from https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio

Expected format: `;` separator, latin-1 encoding, columns `concessionaria`, `mes_ano`,
`sentido`, `praca`, `tipo_cobranca`, `categoria_eixo`, `tipo_de_veiculo`, `volume_total`.

With the file here, `python src/data_volume.py` computes the real average daily traffic of each
EPR toll site and prints it next to the scenarios. The CSV is not committed (it is in
.gitignore) because of its size.
