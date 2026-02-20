# Grid Search - Guia Completo

## 📋 Índice

1. [Introdução](#introdução)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Conceitos Fundamentais](#conceitos-fundamentais)
4. [Uso Detalhado](#uso-detalhado)
5. [Configuração de Hiperparâmetros](#configuração-de-hiperparâmetros)
6. [Execução Paralela](#execução-paralela)
7. [Análise de Resultados](#análise-de-resultados)
8. [Troubleshooting](#troubleshooting)
9. [Boas Práticas](#boas-práticas)
10. [API Reference](#api-reference)

---

## Introdução

O módulo `gridsearch` implementa busca em grade (grid search) para otimização de hiperparâmetros do modelo BERT-PLI. Permite exploração sistemática do espaço de hiperparâmetros para identificar as melhores configurações.

### Características

- ✅ Execução paralela com ProcessPoolExecutor
- ✅ Validação automática de memória RAM
- ✅ Retomada de execuções interrompidas
- ✅ Salvamento incremental de resultados
- ✅ Análise estatística automática
- ✅ Múltiplas interfaces (CLI, Python, PowerShell)

---

## Instalação e Configuração

### Pré-requisitos

```bash
# Dependências Python
pip install psutil  # Monitoramento de memória

# Projeto BERT-PLI já instalado
```

### Estrutura de Arquivos

```
gridsearch/
├── __init__.py                 # Exports do módulo
├── core.py                     # Motor de execução
├── utils.py                    # Utilitários e validações
├── analysis.py                 # Análise de resultados
├── config/
│   ├── grid_search.json        # Grid completo (216 exp)
│   └── grid_search_test.json   # Grid de teste (8 exp)
├── scripts/
│   └── run_grid_search.ps1     # Launcher PowerShell
└── docs/
    ├── GUIDE.md                # Este arquivo
    ├── QUICKSTART.md           # Tutorial rápido
    └── OVERVIEW.md             # Visão técnica
```

---

## Conceitos Fundamentais

### Grid Search

Busca em grade é uma técnica de otimização que:
1. Define conjunto de valores para cada hiperparâmetro
2. Gera todas as combinações possíveis
3. Treina modelo com cada combinação
4. Compara resultados para identificar melhor configuração

**Exemplo:**
```json
{
  "learning_rate": [1e-5, 2e-5],
  "batch_size": [8, 16]
}
```

Gera 4 combinações:
- learning_rate=1e-5, batch_size=8
- learning_rate=1e-5, batch_size=16
- learning_rate=2e-5, batch_size=8
- learning_rate=2e-5, batch_size=16

### Hiperparâmetros Suportados

| Parâmetro | Tipo | Valores Comuns | Impacto |
|-----------|------|----------------|---------|
| `learning_rate` | float | 1e-5, 2e-5, 3e-5 | Velocidade de convergência |
| `batch_size` | int | 8, 16, 32 | Memória e estabilidade |
| `optimizer` | string | "adam", "adamw" | Algoritmo de otimização |
| `dropout` | float | 0.1, 0.2, 0.3 | Regularização |
| `seed` | int | 42, 123, 456 | Reprodutibilidade |

---

## Uso Detalhado

### 1. Interface CLI (Linha de Comando)

#### Teste Rápido

```bash
python -m gridsearch.core \
    --config config/experiments/BertPLI.config \
    --search-config gridsearch/config/grid_search_test.json \
    --parallel 2
```

#### Grid Completo

```bash
python -m gridsearch.core \
    --config config/experiments/BertPLI.config \
    --search-config gridsearch/config/grid_search.json \
    --parallel 2
```

#### Retomar Execução

```bash
python -m gridsearch.core --resume --parallel 2
```

#### Apenas Análise

```bash
python -m gridsearch.core --analyze-only
```

### 2. Interface PowerShell

#### Comandos Básicos

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

#### Personalização

```powershell
# Custom config
.\gridsearch\scripts\run_grid_search.ps1 `
    -Mode test `
    -Config "config/experiments/BertPLI2.config" `
    -SearchConfig "gridsearch/config/custom_grid.json" `
    -Parallel 4
```

### 3. Interface Python (Programática)

```python
from gridsearch import run_grid_search, analyze_results
import json

# Carrega configuração da grade
with open('gridsearch/config/grid_search_test.json') as f:
    config = json.load(f)

# Executa grid search
results = run_grid_search(
    base_config_path='config/experiments/BertPLI.config',
    grid_config=config['hyperparameters'],
    resume=False,
    parallel=2
)

# Analisa resultados
analysis = analyze_results(results)

# Acessa melhores configurações
best_time = analysis['best_by_time']
print(f"Melhor tempo: {best_time['params']}")

best_energy = analysis['best_by_energy']
print(f"Melhor energia: {best_energy['params']}")
```

---

## Configuração de Hiperparâmetros

### Arquivo JSON de Configuração

```json
{
  "description": "Minha busca personalizada",
  "experiment_base": "config/experiments/BertPLI.config",
  "output_dir": "output/experiments/grid_search",
  "parallel_workers": 2,
  
  "hyperparameters": {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "batch_size": [8, 16],
    "optimizer": ["adam", "adamw"],
    "dropout": [0.1, 0.2],
    "seed": [42]
  }
}
```

### Cálculo de Combinações

Total = produto de todas as listas:
```
3 (lr) × 2 (bs) × 2 (opt) × 2 (drop) × 1 (seed) = 24 experimentos
```

### Estimativa de Tempo

```python
from gridsearch.utils import estimate_execution_time

time_hours = estimate_execution_time(
    num_experiments=24,
    avg_time_per_experiment=1800,  # 30 minutos
    parallel_workers=2
)

print(f"Tempo estimado: {time_hours:.1f} horas")
# Saída: Tempo estimado: 6.0 horas
```

---

## Execução Paralela

### Configuração de Workers

```python
# Sequencial (1 worker)
results = run_grid_search(..., parallel=1)

# Paralelo (2 workers) - Recomendado para 32GB RAM
results = run_grid_search(..., parallel=2)

# Paralelo (4 workers) - Requer 64GB+ RAM
results = run_grid_search(..., parallel=4)
```

### Validação de Memória

O módulo valida automaticamente antes da execução:

```python
from gridsearch.utils import check_memory_availability

is_safe, message = check_memory_availability(
    parallel_workers=2,
    max_batch_size=16
)

print(message)
```

**Exemplo de saída:**
```
✓ Memória disponível: 23.9 GB
✓ Estimativa de uso: 7.5 GB
  - Uso por worker: 2.5 GB × 2 = 5.0 GB
  - Sistema operacional: ~2.0 GB
  - Overhead e buffers: ~0.5 GB
✓ Margem de segurança: 16.4 GB
✓ Sistema tem memória suficiente
```

### Recomendações de RAM

| RAM Total | Workers Recomendados | Max Batch Size |
|-----------|---------------------|----------------|
| 16 GB | 1 | 16 |
| 32 GB | 2 | 32 |
| 64 GB | 4 | 32 |
| 128 GB | 8 | 64 |

---

## Análise de Resultados

### Arquivos Gerados

1. **grid_search_results.json** - Resultados completos (JSON)
2. **grid_search_summary.txt** - Resumo legível (TXT)
3. **grid_search_state.json** - Estado para retomada
4. **analysis/full_analysis.json** - Análise estatística completa
5. **analysis/analysis_report.txt** - Relatório detalhado

### Análise Manual

```python
from gridsearch.analysis import (
    compute_descriptive_statistics,
    analyze_by_hyperparameter,
    analyze_correlations,
    rank_configurations
)

# Carregar resultados
import json
with open('output/experiments/grid_search/grid_search_results.json') as f:
    results = json.load(f)

# Estatísticas descritivas
stats = compute_descriptive_statistics(results)
print(f"Tempo médio: {stats['train_time']['mean']:.2f}s")
print(f"Desvio padrão: {stats['train_time']['stdev']:.2f}s")

# Impacto de learning_rate no tempo de treinamento
lr_analysis = analyze_by_hyperparameter(results, 'learning_rate', 'train_time_sec')
for lr, stats in lr_analysis.items():
    print(f"LR={lr}: média={stats['mean']:.2f}s, min={stats['min']:.2f}s")

# Correlações
correlations = analyze_correlations(results)
print(f"Batch size vs RAM: {correlations['batch_size_vs_ram']:.3f}")

# Top 10 configurações
top10 = rank_configurations(results, 
                            metrics=['train_time_sec', 'energy_kwh'],
                            weights=[0.7, 0.3])

for i, config in enumerate(top10, 1):
    print(f"{i}. Exp {config['experiment_idx']}: {config['params']}")
```

### Interpretação dos Resultados

#### Melhor Configuração por Métrica

```python
analysis = analyze_results(results)

# Mais rápido
fastest = analysis['best_by_time']
print(f"Configuração mais rápida:")
print(f"  LR: {fastest['params']['learning_rate']}")
print(f"  Batch: {fastest['params']['batch_size']}")
print(f"  Tempo: {fastest['train_time_sec']}s")

# Mais eficiente energeticamente
efficient = analysis['best_by_energy']
print(f"\nConfiguraçãomais eficiente:")
print(f"  Energia: {efficient['energy_kwh']} kWh")
```

---

## Troubleshooting

### Problema: Out of Memory (OOM)

**Sintomas:**
```
MemoryError: Unable to allocate array
```

**Soluções:**
```powershell
# 1. Reduza workers paralelos
python -m gridsearch.core --config ... --parallel 1

# 2. Reduza batch_size no grid
# Edite gridsearch/config/grid_search.json:
# "batch_size": [8]  # Ao invés de [8, 16, 32]

# 3. Feche outras aplicações
```

### Problema: Execução Interrompida

**Sintomas:**
```
KeyboardInterrupt
ProcessTerminated
```

**Solução:**
```powershell
# Retome de onde parou
python -m gridsearch.core --resume --parallel 2
```

### Problema: Resultados Inconsistentes

**Sintomas:**
- Métricas muito diferentes entre seeds
- Resultados não reproduzíveis

**Soluções:**
```python
# 1. Verifique seeds nos hiperparâmetros
# Certifique-se de usar múltiplos seeds:
"seed": [42, 123, 456]

# 2. Calcule média e desvio padrão
stats = compute_descriptive_statistics(results)
print(f"Desvio padrão: {stats['train_time']['stdev']}")
```

### Problema: Grid Search Muito Lento

**Sintomas:**
- Tempo estimado > 1 semana

**Soluções:**
```json
// 1. Reduza espaço de busca
{
  "learning_rate": [2e-5], // Fixe LR em valor conhecido
  "batch_size": [16, 32],  // Teste apenas 2 valores
  "optimizer": ["adamw"]   // Use apenas o melhor
}

// 2. Use Random Search ao invés de Grid
// (Implemente com sampling aleatório)
```

### Problema: Arquivo de Estado Corrompido

**Sintomas:**
```
json.decoder.JSONDecodeError
```

**Solução:**
```powershell
# Limpe estado e recomece
Remove-Item output/experiments/grid_search/grid_search_state.json
python -m gridsearch.core --config ... --parallel 2
```

---

## Boas Práticas

### 1. Comece Pequeno

```powershell
# SEMPRE teste primeiro com grid pequeno
.\gridsearch\scripts\run_grid_search.ps1 -Mode test
```

### 2. Monitore Recursos

```python
# Use validação de memória
from gridsearch.utils import check_memory_availability

is_safe, msg = check_memory_availability(2, 16)
if not is_safe:
    print("AVISO:", msg)
```

### 3. Salve Configurações

```json
// Documente sua configuração
{
  "description": "Teste de learning rates - 15/02/2026",
  "notes": [
    "Objetivo: Encontrar melhor LR",
    "Dataset: COLIEE 2024",
    "Hardware: RTX 3090, 32GB RAM"
  ],
  "hyperparameters": { ... }
}
```

### 4. Analise Incrementalmente

```python
# Não espere o grid completo terminar
# Analise resultados parciais:
import json

with open('output/experiments/grid_search/grid_search_state.json') as f:
    state = json.load(f)

partial_results = state['results']
print(f"Completados: {len(partial_results)} experimentos")
```

### 5. Use Controle de Versão

```bash
# Salve configurações no Git
git add gridsearch/config/my_grid.json
git commit -m "Grid search: testes de dropout"

# Tag resultados importantes
git tag -a "grid-best-config" -m "Melhor configuração encontrada"
```

---

## API Reference

### Funções Principais

#### `run_grid_search()`

```python
def run_grid_search(
    base_config_path: str,
    grid_config: Dict[str, List[Any]],
    resume: bool = False,
    parallel: int = 1
) -> List[Dict[str, Any]]:
    """
    Executa busca em grade.
    
    Args:
        base_config_path: Caminho do arquivo .config base
        grid_config: Dicionário com hiperparâmetros
        resume: Se True, retoma execução anterior
        parallel: Número de workers paralelos
        
    Returns:
        Lista de dicionários com resultados
    """
```

#### `analyze_results()`

```python
def analyze_results(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analisa resultados e identifica melhores configurações.
    
    Returns:
        {
            'total_experiments': int,
            'successful': int,
            'failed': int,
            'best_by_time': {...},
            'best_by_energy': {...},
            'best_by_ram': {...}
        }
    """
```

#### `check_memory_availability()`

```python
def check_memory_availability(
    parallel_workers: int,
    max_batch_size: int
) -> Tuple[bool, str]:
    """
    Valida se há memória suficiente.
    
    Returns:
        (is_safe, message)
    """
```

### Classes e Estruturas

#### Resultado de Experimento

```python
{
    "grid_experiment_idx": 0,
    "grid_params": {
        "learning_rate": 1e-5,
        "batch_size": 16,
        "optimizer": "adam"
    },
    "status": "success",  # ou "failed"
    "resources": {
        "train_time_sec": 1234.56,
        "energy_kwh": 0.0234,
        "peak_ram_mb": 8192.0
    }
}
```

---

## Conclusão

Este guia cobre todos os aspectos do módulo `gridsearch`. Para mais informações:

- **Tutorial Rápido:** `gridsearch/docs/QUICKSTART.md`
- **Visão Técnica:** `gridsearch/docs/OVERVIEW.md`
- **Código Fonte:** `gridsearch/core.py`, `gridsearch/analysis.py`

**Dúvidas?** Consulte o código-fonte ou abra uma issue no repositório.

---

**Versão:** 1.0.0  
**Última atualização:** 15/02/2026  
**Autores:** BERT-PLI Team
