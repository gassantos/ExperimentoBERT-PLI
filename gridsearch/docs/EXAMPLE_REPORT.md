# Exemplo de Relatório Grid Search

Este documento demonstra o formato de saída do relatório aprimorado do Grid Search.

## Relatório Gerado (`grid_search_summary.txt`)

```
================================================================================
GRID SEARCH - RELATÓRIO DE RESULTADOS
================================================================================
Data: 19/02/2026 14:35:20

RESUMO GERAL:
  Total de experimentos: 216
  Bem-sucedidos: 210
  Falhos: 6

MELHOR CONFIGURAÇÃO (Tempo de Treinamento):
  Experimento: 42
  Tempo: 1253.45 segundos
  Parâmetros:
    - learning_rate: 5e-05
    - batch_size: 32
    - optimizer: adamw
    - dropout: 0.1
    - seed: 42

MELHOR CONFIGURAÇÃO (Eficiência Energética):
  Experimento: 87
  Energia: 0.0785 kWh
  Parâmetros:
    - learning_rate: 1e-05
    - batch_size: 8
    - optimizer: adam
    - dropout: 0.2
    - seed: 123

MELHOR CONFIGURAÇÃO (Uso de Memória RAM):
  Experimento: 15
  RAM Pico: 5834 MB
  Parâmetros:
    - learning_rate: 2e-05
    - batch_size: 8
    - optimizer: adam
    - dropout: 0.1
    - seed: 42

MELHOR CONFIGURAÇÃO (Menor Emissão de Carbono):
  Experimento: 87
  Emissão CO2: 0.012432 kg
  Parâmetros:
    - learning_rate: 1e-05
    - batch_size: 8
    - optimizer: adam
    - dropout: 0.2
    - seed: 123

MELHOR CONFIGURAÇÃO (Menor Custo Financeiro):
  Experimento: 87
  Custo: $0.0094 USD
  (Tarifa: $0.1200/kWh)
  Parâmetros:
    - learning_rate: 1e-05
    - batch_size: 8
    - optimizer: adam
    - dropout: 0.2
    - seed: 123

================================================================================

ESTATÍSTICAS GERAIS DOS EXPERIMENTOS BEM-SUCEDIDOS:

  Tempo total de treinamento: 287450.34 segundos (79.85 horas)
  Energia total consumida: 18.4523 kWh
  Emissão total de CO2: 2.918765 kg (2918.76 g)
  Custo financeiro total: $2.2143 USD

================================================================================
```

## Interpretação dos Resultados

### 🎯 Trade-offs Identificados

O relatório revela trade-offs importantes:

1. **Velocidade vs. Eficiência Energética**
   - Configuração mais rápida: batch_size=32, lr=5e-5
   - Configuração mais eficiente: batch_size=8, lr=1e-5
   - Trade-off: 60% mais rápido, mas consome 85% mais energia

2. **Custo vs. Performance**
   - Configuração mais barata coincide com menor emissão de CO2
   - Batch sizes menores são mais eficientes em termos de custo/carbono
   - Learning rates menores reduzem tempo de convergência

### 💰 Análise de Custos

Com tarifa de $0.12/kWh:
- **Experimento único**: ~$0.01 USD
- **Grid completo (216 exp)**: ~$2.21 USD
- **Economia potencial**: Escolhendo configuração ótima, pode-se reduzir custos em 40-50%

### 🌍 Análise Ambiental

- **CO2 total**: 2.92 kg ≈ dirigir 12 km em carro médio
- **Melhor configuração**: 0.012 kg CO2 (12g)
- **Pior configuração**: 0.021 kg CO2 (21g)
- **Diferença**: 75% de redução escolhendo configuração ótima

### 🏆 Configuração Recomendada

Baseado em análise multi-critério (balanceando todas métricas):

```python
{
  "learning_rate": 1e-05,
  "batch_size": 8,
  "optimizer": "adam",
  "dropout": 0.2,
  "seed": 123
}
```

**Justificativa:**
- ✅ Menor emissão de carbono
- ✅ Menor custo financeiro
- ✅ Menor consumo energético
- ✅ Uso moderado de RAM
- ⚠️ Tempo de treinamento 40% maior (trade-off aceitável)

## 📊 Dados Estruturados (JSON)

O arquivo `grid_search_results.json` contém todos os detalhes:

```json
{
  "timestamp": "2026-02-19T14:35:20",
  "total_experiments": 216,
  "successful": 210,
  "failed": 6,
  "energy_cost_usd_per_kwh": 0.12,
  
  "best_by_carbon": {
    "experiment_idx": 87,
    "params": {
      "learning_rate": 1e-05,
      "batch_size": 8,
      "optimizer": "adam",
      "dropout": 0.2,
      "seed": 123
    },
    "emissions_kg_co2": 0.012432
  },
  
  "best_by_cost": {
    "experiment_idx": 87,
    "params": {
      "learning_rate": 1e-05,
      "batch_size": 8,
      "optimizer": "adam",
      "dropout": 0.2,
      "seed": 123
    },
    "cost_usd": 0.0094
  },
  
  "all_results": [...]
}
```

## 🔧 Configurando Tarifa Regional

Diferentes regiões têm tarifas diferentes:

```bash
# Brasil (média R$0.60/kWh ≈ $0.12/kWh)
export ENERGY_COST_USD_PER_KWH=0.12

# EUA/Califórnia (~$0.25/kWh)
export ENERGY_COST_USD_PER_KWH=0.25

# Europa/Alemanha (~$0.35/kWh)
export ENERGY_COST_USD_PER_KWH=0.35

# Cloud Computing (AWS p3.2xlarge: ~$3.06/h, 70W média ≈ $0.40/kWh)
export ENERGY_COST_USD_PER_KWH=0.40
```

## 📈 Visualizações Recomendadas

Para análise visual adicional, use:

```python
from gridsearch.analysis import (
    plot_pareto_front,
    plot_hyperparameter_impact,
    generate_heatmap
)

# Fronteira de Pareto: Custo vs. Tempo
plot_pareto_front(
    results,
    x_metric="train_time_sec",
    y_metric="cost_usd",
    output_path="output/pareto_cost_vs_time.png"
)

# Impacto do batch size no CO2
plot_hyperparameter_impact(
    results,
    param_name="batch_size",
    metric_name="emissions_kg_co2",
    output_path="output/batch_vs_carbon.png"
)

# Heatmap: Learning rate vs. Batch size (Custo)
generate_heatmap(
    results,
    x_param="learning_rate",
    y_param="batch_size",
    metric="cost_usd",
    output_path="output/heatmap_cost.png"
)
```

## 🎓 Lições Aprendidas

1. **Batch size** tem maior impacto em consumo energético e custo
2. **Learning rate** menor reduz desperdício computacional (menos épocas para convergir)
3. **Optimizer AdamW** tende a ser 5-10% mais rápido que Adam, mas consome mais RAM
4. **Dropout** tem impacto mínimo em eficiência energética
5. **Seeds diferentes** produzem variação de ±15% em todas métricas

## 📚 Referências

- [CodeCarbon Documentation](https://codecarbon.io/)
- [Green AI Research](https://arxiv.org/abs/1907.10597)
- [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243)
