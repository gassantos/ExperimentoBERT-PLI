# Grid Search - Visão Técnica

## Arquitetura do Módulo

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                      GRIDSEARCH MODULE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  core.py     │  │  utils.py    │  │  analysis.py    │  │
│  │              │  │              │  │                 │  │
│  │ • Execution  │  │ • Memory     │  │ • Statistics    │  │
│  │ • Iteration  │  │ • Validation │  │ • Correlations  │  │
│  │ • State Mgmt │  │ • Estimation │  │ • Ranking       │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                 │                    │           │
│         └─────────────────┴────────────────────┘           │
│                           │                                │
│                  ┌────────▼────────┐                       │
│                  │  __init__.py    │                       │
│                  │  (Public API)   │                       │
│                  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Fluxo de Execução

### 1. Inicialização

```python
# 1. Carrega configurações
base_config = ConfigParser.read('BertPLI.config')
grid_config = json.load('grid_search.json')

# 2. Valida hiperparâmetros
validate_grid_config(grid_config)

# 3. Gera combinações
param_grid = generate_parameter_grid(grid_config)
# Resultado: [{lr:1e-5, bs:8}, {lr:1e-5, bs:16}, ...]

# 4. Valida memória
check_memory_availability(parallel=2, batch_size=16)
```

### 2. Execução

#### Modo Sequencial (parallel=1)

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Exp 1  │ --> │  Exp 2  │ --> │  Exp 3  │ --> │  Exp 4  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
  1800s           1800s           1800s           1800s
  
Total: 7200s = 2 horas
```

#### Modo Paralelo (parallel=2)

```
┌─────────┐     ┌─────────┐
│  Exp 1  │     │  Exp 3  │
└─────────┘     └─────────┘
     \             /
      \           /
       v         v
    ProcessPoolExecutor
       /           \
      /             \
     v               v
┌─────────┐     ┌─────────┐
│  Exp 2  │     │  Exp 4  │
└─────────┘     └─────────┘

Total: 3600s = 1 hora
```

### 3. ProcessPoolExecutor

```python
with ProcessPoolExecutor(max_workers=2) as executor:
    # Submete tarefas
    futures = {
        executor.submit(run_single_experiment, idx, cfg, params): idx
        for idx, cfg, params in pending_experiments
    }
    
    # Aguarda conclusão
    for future in as_completed(futures):
        result = future.result()
        save_state(completed, results)  # Salva incrementalmente
```

**Importante:** Cada worker spawna subprocess `uv run python train.py`

### 4. Salvamento de Estado

```python
# Estado salvo após CADA experimento
state = {
    "timestamp": "2026-02-15T14:30:00",
    "completed_experiments": [0, 1, 2],  # Índices concluídos
    "results": [
        {"grid_experiment_idx": 0, ...},
        {"grid_experiment_idx": 1, ...},
        {"grid_experiment_idx": 2, ...}
    ]
}

# Permite retomada com --resume
```

## Modelo de Memória

### Consumo por Componente

```
┌────────────────────────────────────────────────────────┐
│ Total System RAM: 32 GB                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Sistema Operacional:              ~2.0 GB             │
│                                                        │
│ ┌────────────────────────────────────────────────┐   │
│ │ ProcessPoolExecutor (1 main + 2 workers)       │   │
│ │                                                │   │
│ │ Main Process:                 ~0.5 GB          │   │
│ │                                                │   │
│ │ Worker 1 (BERT training):                      │   │
│ │   • Model (BERT-base):         0.4 GB          │   │
│ │   • Optimizer state:           0.4 GB          │   │
│ │   • Gradients:                 0.4 GB          │   │
│ │   • Activations (batch=16):    0.8 GB          │   │
│ │   • DataLoader cache:          0.5 GB          │   │
│ │   Subtotal:                   ~2.5 GB          │   │
│ │                                                │   │
│ │ Worker 2 (BERT training):                      │   │
│ │   Subtotal:                   ~2.5 GB          │   │
│ │                                                │   │
│ │ Total Workers:                 5.0 GB          │   │
│ └────────────────────────────────────────────────┘   │
│                                                        │
│ Overhead e Buffers:               ~0.5 GB             │
│                                                        │
│ TOTAL USADO:                      ~8.0 GB             │
│                                                        │
│ LIVRE/CACHE:                     ~24.0 GB             │
└────────────────────────────────────────────────────────┘
```

### Fórmula de Estimativa

```python
def estimate_memory_requirements(parallel_workers, max_batch_size):
    # Memória base por processo de treinamento
    bert_model_mb = 440  # BERT-base fp16
    optimizer_mb = 440   # Adam state
    gradients_mb = 440   # Backprop
    
    # Memória dependente de batch
    activation_per_sample = 50  # MB
    activations_mb = max_batch_size * activation_per_sample
    
    # DataLoader
    dataloader_mb = 512
    
    # Total por worker
    per_worker_mb = (
        bert_model_mb +
        optimizer_mb +
        gradients_mb +
        activations_mb +
        dataloader_mb
    )
    
    # Total do sistema
    system_mb = 2048
    main_process_mb = 512
    overhead_mb = 512
    
    total_mb = (
        system_mb +
        main_process_mb +
        (per_worker_mb * parallel_workers) +
        overhead_mb
    )
    
    return total_mb / 1024  # Converte para GB
```

## Análise de Resultados

### Pipeline de Análise

```
Raw Results (JSON)
        │
        ▼
┌───────────────────┐
│ Filter Successful │ --> Filtra experimentos com status="success"
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Compute Statistics│ --> Média, mediana, desvio padrão
└────────┬──────────┘
         │
         ├──> Train Time Stats
         ├──> Energy Stats
         └──> RAM Stats
         │
         ▼
┌───────────────────┐
│ Analyze by Param  │ --> Agrupa por hiperparâmetro
└────────┬──────────┘
         │
         ├──> Learning Rate Impact
         ├──> Batch Size Impact
         └──> Optimizer Impact
         │
         ▼
┌───────────────────┐
│ Compute Corr.     │ --> Pearson correlation
└────────┬──────────┘
         │
         ├──> LR vs Time
         ├──> Batch vs RAM
         └──> Dropout vs Energy
         │
         ▼
┌───────────────────┐
│ Rank Configs      │ --> Multi-objective ranking
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Generate Report   │ --> Texto formatado
└───────────────────┘
```

### Algoritmo de Ranking

```python
# 1. Normalização Min-Max
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

# 2. Score Ponderado
def compute_score(metrics, weights):
    # metrics = [norm_time, norm_energy, norm_ram]
    # weights = [0.5, 0.3, 0.2]
    
    score = sum(m * w for m, w in zip(metrics, weights))
    return score

# 3. Ordenação (menor score = melhor)
ranked = sorted(configs, key=lambda x: x['score'])
```

## Estrutura de Dados

### Configuração de Grid

```python
{
    "description": str,
    "experiment_base": str,  # Path to .config
    "output_dir": str,
    "parallel_workers": int,
    
    "hyperparameters": {
        "learning_rate": List[float],
        "batch_size": List[int],
        "optimizer": List[str],
        "dropout": List[float],
        "seed": List[int]
    },
    
    "notes": List[str],  # Opcional
    "recommendations": Dict[str, Any]  # Opcional
}
```

### Resultado de Experimento

```python
{
    "grid_experiment_idx": int,      # Índice na grade
    "grid_params": {                 # Hiperparâmetros usados
        "learning_rate": float,
        "batch_size": int,
        "optimizer": str,
        ...
    },
    "status": "success" | "failed",
    
    # Se success:
    "resources": {
        "train_time_sec": float,     # Segundos
        "energy_kwh": float,          # kWh consumidos
        "peak_ram_mb": float,         # MB pico
        "gpu_util_avg": float,        # % utilização GPU
        ...
    },
    
    # Se failed:
    "error": str,                     # Mensagem de erro
    "traceback": str                  # Stack trace completo
}
```

### Estado de Execução

```python
{
    "timestamp": str,                 # ISO 8601
    "completed_experiments": List[int],  # Índices concluídos
    "results": List[Dict],            # Resultados completos
    
    # Metadados (opcional)
    "total_experiments": int,
    "grid_config_hash": str,          # Hash da configuração
    "resume_count": int               # Quantas vezes foi retomado
}
```

## Performance

### Benchmarks

#### Sistema: 32GB RAM, RTX 3090, i9-12900K

| Config | Experimentos | Tempo Total | Tempo/Exp | RAM Pico |
|--------|--------------|-------------|-----------|----------|
| Test (parallel=1) | 8 | 4h 00m | 30min | 3.2 GB |
| Test (parallel=2) | 8 | 2h 10m | 16min | 6.8 GB |
| Full (parallel=1) | 216 | 108h (4.5d) | 30min | 3.2 GB |
| Full (parallel=2) | 216 | 58h (2.4d) | 16min | 6.8 GB |
| Full (parallel=4) | 216 | 32h (1.3d) | 9min | 13.5 GB |

**Observação:** parallel=4 requer 64GB+ RAM

### Otimizações Implementadas

1. **Salvamento Incremental**
   - Estado salvo após cada experimento
   - Evita perda total em caso de falha
   
2. **ProcessPoolExecutor**
   - Paralelismo real (não threads)
   - Bypass do GIL do Python
   
3. **Lazy Loading**
   - Configurações carregadas apenas quando necessário
   - Reduz overhead de memória
   
4. **Validação Antecipada**
   - Memória validada ANTES da execução
   - Evita OOM após horas de processamento

## Limitações Conhecidas

### 1. Espaço de Busca Exponencial

```python
# Cuidado com explosão combinatória
grid = {
    "param1": [1, 2, 3, 4, 5],        # 5
    "param2": [1, 2, 3, 4, 5],        # 5
    "param3": [1, 2, 3, 4, 5],        # 5
    "param4": [1, 2, 3, 4, 5],        # 5
    "param5": [1, 2, 3, 4, 5]         # 5
}

# Total: 5^5 = 3125 experimentos!
```

**Solução:** Use Random Search ou Bayesian Optimization

### 2. Sem Early Stopping

Grid search atual **não** implementa early stopping de experimentos ruins.

**Workaround:**
```python
# Analise resultados parciais
# Interrompa manualmente com Ctrl+C
# Remova configurações ruins do grid
# Retome com --resume
```

### 3. Apenas Hiperparâmetros Discretos

Não suporta distribuições contínuas ou condicionais.

**Exemplo não suportado:**
```python
# ❌ Condicional: se optimizer=adam, então beta1=[0.9, 0.95]
# ❌ Contínuo: learning_rate ~ LogUniform(1e-6, 1e-3)
```

## Extensões Futuras

### 1. Random Search

```python
def random_search(grid_config, n_samples=50):
    """Amostra aleatória ao invés de grid completo"""
    # TODO: Implementar
    pass
```

### 2. Bayesian Optimization

```python
from skopt import gp_minimize

def bayesian_search(objective, space):
    """Otimização Bayesiana com Gaussian Processes"""
    # TODO: Integrar scikit-optimize
    pass
```

### 3. Multi-GPU Support

```python
def run_grid_search_multigpu(grid_config, gpus=[0, 1, 2, 3]):
    """Distribui experimentos por múltiplas GPUs"""
    # TODO: Implementar com CUDA_VISIBLE_DEVICES
    pass
```

### 4. Dashboarding Tempo-Real

```python
def start_dashboard(port=8050):
    """Dashboard Plotly Dash para monitoramento"""
    # TODO: Implementar visualização em tempo real
    pass
```

## Referências

- **Grid Search Theory:** Bergstra & Bengio (2012) - "Random Search for Hyper-Parameter Optimization"
- **ProcessPoolExecutor:** Python concurrent.futures documentation
- **BERT Memory:** Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"

---

**Versão:** 1.0.0  
**Última atualização:** 15/02/2026  
**Autores:** BERT-PLI Team
