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

```sh
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

### Critérios de Análise

O módulo identifica as melhores configurações por **5 critérios diferentes**:

1. **⏱️ Tempo de Treinamento** (`train_time_sec`)
   - Menor tempo é melhor
   - Métrica: segundos

2. **⚡ Eficiência Energética** (`energy_kwh`)
   - Menor consumo é melhor
   - Métrica: kWh (quilowatt-hora)

3. **🧠 Uso de Memória RAM** (`peak_ram_mb`)
   - Menor uso é melhor
   - Métrica: MB (megabytes)

4. **🌍 Emissão de Carbono** (`emissions_kg_co2`)
   - Menor emissão é melhor
   - Métrica: kg CO₂
   - Calculado via CodeCarbon

5. **💰 Custo Financeiro** (`cost_usd`)
   - Menor custo é melhor
   - Métrica: USD (dólares americanos)
   - Calculado: `energy_kwh × tarifa_energia`
   - Tarifa padrão: $0.12/kWh (configurável)

### Configurando a Tarifa de Energia

```powershell
# Windows PowerShell
$env:ENERGY_COST_USD_PER_KWH = "0.15"  # $0.15 por kWh
python -m gridsearch.core --config ... --parallel 2

# Linux/WSL
export ENERGY_COST_USD_PER_KWH=0.15
python -m gridsearch.core --config ... --parallel 2
```

### Análise Manual

```python
from gridsearch.analysis import (
    compute_descriptive_statistics,
    analyze_correlations,
    rank_configurations
)

# Estatísticas descritivas de todas as métricas
stats = compute_descriptive_statistics(results)
print(f"Tempo médio: {stats['train_time']['mean']:.2f}s")
print(f"CO2 médio: {stats['emissions_kg_co2']['mean']:.6f} kg")
print(f"Custo médio: ${stats['cost_usd']['mean']:.4f}")

# Correlações entre hiperparâmetros e métricas
corr = analyze_correlations(results)

# Ranking multi-critério (exemplo: 40% tempo, 30% CO2, 30% custo)
top10 = rank_configurations(
    results,
    metrics=["train_time_sec", "emissions_kg_co2", "cost_usd"],
    weights=[0.4, 0.3, 0.3]
)[:10]

# Análise por hiperparâmetro específico
from gridsearch.analysis import analyze_by_hyperparameter

batch_impact_on_carbon = analyze_by_hyperparameter(
    results,
    param_name="batch_size",
    metric_name="emissions_kg_co2"
)

batch_impact_on_cost = analyze_by_hyperparameter(
    results,
    param_name="batch_size",
    metric_name="cost_usd"
)
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

```yaml
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
