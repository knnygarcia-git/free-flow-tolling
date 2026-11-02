# Data

| Folder | Contents | Source | License |
|---|---|---|---|
| `raw/sem_parar/` | Registry of Free Flow gantries, highways, concessionaires, accepted tags, legal basis and timeline in Brazil (updated 2026-08-26). Files and columns are in Portuguese, as published. | [Sem-Parar/SEM-PARAR-FREE-FLOW](https://github.com/Sem-Parar/SEM-PARAR-FREE-FLOW), which compiles the registries of ANTT and the concessionaires | CC BY 4.0 (© 2026 Sem Parar Instituição de Pagamento Ltda.) |
| `raw/antt/` | (optional) Traffic volume per toll site, from ANTT | [dados.antt.gov.br](https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio) | CC BY |
| `synthetic/` | Synthetic passes for the ML experiments, created by `src/ml/generate_events.py` (not in git; recreated in seconds) | this project | |

Files in `raw/` are never edited; all cleaning happens in the scripts in `src/`.
