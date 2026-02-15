# Grid Search Module - BERT-PLI

Módulo Python profissional para execução de busca em grade de hiperparâmetros.

## 🚀 Instalação

O módulo já está integrado ao projeto. Nenhuma instalação adicional é necessária.

## 📖 Uso Rápido

### Modo CLI

```powershell
# Teste rápido (8 experimentos)
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search_test.json \
                          --parallel 2

# Execução completa (216 experimentos)
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search.json \
                          --parallel 2

# Retomar execução interrompida
python -m gridsearch.core --resume --parallel 2

# Análise de resultados
python -m gridsearch.core --analyze-only
```

### Modo PowerShell

```powershell
# Teste
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 2

# Completo
.\gridsearch\scripts\run_grid_search.ps1 -Mode full -Parallel 2

# Retomar
.\gridsearch\scripts\run_grid_search.ps1 -Mode resume -Parallel 2

# Análise
.\gridsearch\scripts\run_grid_search.ps1 -Mode analyze
```

### Modo Python (importável)

```python
from gridsearch import run_grid_search
import json

# Carrega configuração
with open('gridsearch/config/grid_search_test.json') as f:
    grid_config = json.load(f)

# Executa
results = run_grid_search(
    base_config_path='config/experiments/BertPLI.config',
    grid_config=grid_config['hyperparameters'],
    parallel=2
)

# Analisa
from gridsearch import analyze_results
analysis = analyze_results(results)
```

## 📁 Estrutura

```
gridsearch/
├── __init__.py              # Exports do módulo
├── core.py                  # Motor de execução
├── utils.py                 # Validações de memória
├── analysis.py              # Análise de resultados
├── config/
│   ├── grid_search.json     # Grid completo (216 exp)
│   └── grid_search_test.json # Grid de teste (8 exp)
├── scripts/
│   └── run_grid_search.ps1  # Launcher PowerShell
└── docs/
    ├── GUIDE.md             # Guia completo
    ├── QUICKSTART.md        # Início rápido
    └── OVERVIEW.md          # Visão geral
```

## ⚙️ Configurações

### Grid de Teste (8 experimentos)
- Learning rates: [1e-5, 2e-5]
- Batch sizes: [8, 16]
- Otimizadores: ["adam"]
- **Tempo:** ~2-3 horas com parallel=2
- **Memória:** ~7-10 GB

### Grid Completo (216 experimentos)
- Learning rates: [1e-5, 2e-5, 3e-5, 5e-5]
- Batch sizes: [8, 16, 32]
- Otimizadores: ["adam", "adamw"]
- Dropouts: [0.1, 0.2, 0.3]
- Seeds: [42, 123, 456]
- **Tempo:** ~72-108 horas com parallel=2
- **Memória:** 32GB RAM recomendado

## 🔍 Análise de Resultados

Os resultados são salvos automaticamente em:

- `output/experiments/grid_search/grid_search_results.json` - Resultados completos
- `output/experiments/grid_search/grid_search_summary.txt` - Resumo legível
- `output/experiments/grid_search/analysis/` - Análises detalhadas

### Análise Manual

```python
from gridsearch.analysis import (
    compute_descriptive_statistics,
    analyze_correlations,
    rank_configurations
)

# Estatísticas
stats = compute_descriptive_statistics(results)

# Correlações
corr = analyze_correlations(results)

# Ranking
top10 = rank_configurations(results)[:10]
```

## 🛡️ Validação de Memória

O módulo valida automaticamente disponibilidade de RAM antes da execução:

```python
from gridsearch.utils import check_memory_availability

is_safe, message = check_memory_availability(
    parallel_workers=2,
    max_batch_size=16
)

print(message)
```

Exemplo de saída:
```
✓ Memória disponível: 23.9 GB
✓ Estimativa de uso: 7.5 GB
✓ Margem de segurança: 16.4 GB
✓ Sistema tem memória suficiente
```

## 🚨 Alertas Importantes

1. **Memória:** Com 32GB RAM, use `--parallel 2` no máximo
2. **Tempo:** Grid completo pode levar 3-4 dias
3. **Retomada:** Use `--resume` para continuar execuções interrompidas
4. **Backup:** Resultados parciais são salvos incrementalmente

## 📚 Documentação Completa

Consulte os arquivos em `gridsearch/docs/`:

- **GUIDE.md** - Guia detalhado de uso
- **QUICKSTART.md** - Tutorial de 5 minutos
- **OVERVIEW.md** - Visão técnica do módulo

## 🆘 Troubleshooting

### OOM (Out of Memory)

```powershell
# Reduza o paralelismo
python -m gridsearch.core --config ... --parallel 1
```

### Execução Interrompida

```powershell
# Retome de onde parou
python -m gridsearch.core --resume --parallel 2
```

### Resultados Corrompidos

```powershell
# Force nova execução
Remove-Item output/experiments/grid_search/grid_search_state.json
python -m gridsearch.core --config ... --parallel 2
```

## 📄 Licença

Parte do projeto BERT-PLI.

## 👥 Autores

BERT-PLI Team - 2026
