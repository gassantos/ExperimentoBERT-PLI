# Guia Rápido: Análise de Emissões e Custos

## 🎯 Objetivo

Este guia mostra como utilizar as novas funcionalidades de análise de **Emissão de Carbono** e **Custo Financeiro** no Grid Search do BERT-PLI.

## 🆕 Novas Métricas

### 1. Emissão de Carbono (`emissions_kg_co2`)

- **Unidade**: kg CO₂
- **Fonte**: CodeCarbon tracker
- **Interpretação**: Menor é melhor
- **Cálculo**: Automático durante treinamento

### 2. Custo Financeiro (`cost_usd`)

- **Unidade**: USD (dólares)
- **Fórmula**: `energy_kwh × tarifa_usd_per_kwh`
- **Interpretação**: Menor é melhor
- **Configuração**: Via variável de ambiente

## ⚙️ Configuração

### Definir Tarifa de Energia

**Windows PowerShell:**

```powershell
$env:ENERGY_COST_USD_PER_KWH = "0.12"
```

**Linux/WSL:**

```bash
export ENERGY_COST_USD_PER_KWH=0.12
```

**Python:**

```python
import os
os.environ["ENERGY_COST_USD_PER_KWH"] = "0.12"
```

### Tarifas de Referência

| Localização       | Tarifa (USD/kWh) |
|-------------------|------------------|
| Brasil (média)    | $0.12            |
| EUA (média)       | $0.14            |
| EUA (Califórnia)  | $0.25            |
| Europa (média)    | $0.28            |
| Europa (Alemanha) | $0.35            |
| AWS p3.2xlarge    | $0.40            |
| Google Cloud V100 | $0.45            |

## 🚀 Uso Básico

### Executar Grid Search

```powershell
# Com tarifa padrão ($0.12/kWh)
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search_test.json \
                          --parallel 2

# Com tarifa personalizada
$env:ENERGY_COST_USD_PER_KWH = "0.25"
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search_test.json \
                          --parallel 2
```

### Analisar Resultados

```powershell
python -m gridsearch.core --analyze-only
```

## 📊 Interpretando o Relatório

### Seções do Relatório

O relatório agora inclui **5 critérios de otimalidade**:

1. ⏱️ **Tempo de Treinamento** - Velocidade de execução
2. ⚡ **Eficiência Energética** - Consumo em kWh
3. 🧠 **Uso de Memória RAM** - Pico de RAM usado
4. 🌍 **Emissão de Carbono** - kg CO₂ emitidos
5. 💰 **Custo Financeiro** - Custo em USD

### Estatísticas Gerais

Ao final do relatório, você verá:

```yaml
ESTATÍSTICAS GERAIS DOS EXPERIMENTOS BEM-SUCEDIDOS:

  Tempo total de treinamento: 287450.34 segundos (79.85 horas)
  Energia total consumida: 18.4523 kWh
  Emissão total de CO2: 2.918765 kg (2918.76 g)
  Custo financeiro total: $2.2143 USD
```

**Interpretação:**

- **Tempo total**: Duração acumulada de todos experimentos
- **Energia**: Consumo elétrico total
- **CO₂**: Equivalente em kg de dióxido de carbono
- **Custo**: Valor monetário baseado na tarifa configurada

## 🔬 Análise Programática

### Estatísticas Descritivas

```python
from gridsearch.analysis import compute_descriptive_statistics
import json

# Carregar resultados
with open('output/experiments/grid_search/grid_search_results.json') as f:
    results = json.load(f)['all_results']

# Calcular estatísticas
stats = compute_descriptive_statistics(results)

print(f"CO2 Médio: {stats['emissions_kg_co2']['mean']:.6f} kg")
print(f"CO2 Mínimo: {stats['emissions_kg_co2']['min']:.6f} kg")
print(f"CO2 Máximo: {stats['emissions_kg_co2']['max']:.6f} kg")
print(f"Desvio Padrão: {stats['emissions_kg_co2']['stdev']:.6f} kg")

print(f"\nCusto Médio: ${stats['cost_usd']['mean']:.4f}")
print(f"Custo Mínimo: ${stats['cost_usd']['min']:.4f}")
print(f"Custo Máximo: ${stats['cost_usd']['max']:.4f}")
```

### Encontrar Melhor Configuração

```python
# Melhor por emissão
best_carbon = min(
    [r for r in results if r['status'] == 'success'],
    key=lambda x: x['resources']['emissions_kg_co2']
)

print(f"Menor emissão: {best_carbon['resources']['emissions_kg_co2']:.6f} kg")
print(f"Parâmetros: {best_carbon['grid_params']}")

# Melhor por custo
best_cost = min(
    [r for r in results if r['status'] == 'success'],
    key=lambda x: x['resources']['cost_usd']
)

print(f"\nMenor custo: ${best_cost['resources']['cost_usd']:.4f}")
print(f"Parâmetros: {best_cost['grid_params']}")
```

### Ranking Multi-Critério

```python
from gridsearch.analysis import rank_configurations

# Exemplo 1: Priorizar custo e carbono igualmente
top10_eco = rank_configurations(
    results,
    metrics=["emissions_kg_co2", "cost_usd"],
    weights=[0.5, 0.5]  # 50% cada
)[:10]

# Exemplo 2: Balancear tempo, custo e carbono
top10_balanced = rank_configurations(
    results,
    metrics=["train_time_sec", "emissions_kg_co2", "cost_usd"],
    weights=[0.4, 0.3, 0.3]  # 40% tempo, 30% carbono, 30% custo
)[:10]

# Exemplo 3: Priorizar apenas sustentabilidade
top10_green = rank_configurations(
    results,
    metrics=["emissions_kg_co2", "energy_kwh"],
    weights=[0.6, 0.4]  # 60% carbono, 40% energia
)[:10]

for rank, config in enumerate(top10_green, 1):
    print(f"{rank}. Exp {config['experiment_idx']}: score={config['score']:.4f}")
    print(f"   Parâmetros: {config['params']}")
```

### Análise por Hiperparâmetro

```python
from gridsearch.analysis import analyze_by_hyperparameter

# Impacto do batch_size no CO2
batch_vs_carbon = analyze_by_hyperparameter(
    results,
    param_name="batch_size",
    metric_name="emissions_kg_co2"
)

print("Impacto do Batch Size nas Emissões:")
for batch_size, stats in sorted(batch_vs_carbon.items()):
    print(f"  batch_size={batch_size}: {stats['mean']:.6f} kg CO2 (±{stats['stdev']:.6f})")

# Impacto do learning_rate no custo
lr_vs_cost = analyze_by_hyperparameter(
    results,
    param_name="learning_rate",
    metric_name="cost_usd"
)

print("\nImpacto do Learning Rate no Custo:")
for lr, stats in sorted(lr_vs_cost.items()):
    print(f"  lr={lr}: ${stats['mean']:.4f} (±${stats['stdev']:.4f})")
```

## 💡 Dicas Práticas

### 1. Reduzir Custos

- ✅ Use batch sizes menores (8 ou 16)
- ✅ Escolha learning rates menores (1e-5 ou 2e-5)
- ✅ Prefira Adam ao AdamW (ligeiramente mais eficiente)
- ✅ Execute treinamento em horários de tarifa reduzida

### 2. Reduzir Emissões

- ✅ Mesmas recomendações de redução de custos
- ✅ Use infraestrutura com energia renovável
- ✅ Considere cloud providers com certificação verde
- ✅ Evite re-treinar modelos desnecessariamente

### 3. Comparar Experimentos

```python
# Calcular economia ao escolher melhor configuração
worst = max(results, key=lambda x: x['resources']['cost_usd'])
best = min(results, key=lambda x: x['resources']['cost_usd'])

savings = worst['resources']['cost_usd'] - best['resources']['cost_usd']
savings_pct = (savings / worst['resources']['cost_usd']) * 100

print(f"Economia: ${savings:.4f} ({savings_pct:.1f}%)")

# Extrapolar para produção
n_trainings_per_year = 100
annual_savings = savings * n_trainings_per_year

print(f"Economia anual (100 treinamentos): ${annual_savings:.2f}")
```

## 📈 Benchmarks Típicos

### Modelo BERT-base

| Métrica   | Típico    | Ótimo     | Ruim      |
|-----------|-----------|-----------|-----------|
| Tempo     | 1800s     | 1200s     | 2500s     |
| Energia   | 0.12 kWh  | 0.08 kWh  | 0.18 kWh  |
| CO₂       | 0.019 kg  | 0.012 kg  | 0.028 kg  |
| Custo     | $0.014    | $0.010    | $0.022    |

### Grid Search Completo (216 experimentos)

| Métrica       | Estimativa    |
|---------------|------------   |
| Tempo total   | 72-108 horas  |
| Energia total | 15-25 kWh     |
| CO₂ total     | 2.4-4.0 kg    |
| Custo total¹  | $1.80-$3.00   |

¹o custo do KWh é, em média, $0.12/KWh

## 🆘 Troubleshooting

### Problema: Métricas de CO₂ retornam `null`

**Solução:** Verifique se CodeCarbon está instalado e habilitado:

```bash
pip show codecarbon
```

Verifique configuração em `config/experiments/BertPLI.config`:

```ini
[monitoring]
enable_monitoring = true
```

### Problema: Custo calculado parece incorreto

**Solução:** Verifique tarifa configurada:

```python
import os
print(os.environ.get('ENERGY_COST_USD_PER_KWH', 'Não configurada'))
```

### Problema: Análise não mostra novas métricas

**Solução:** Re-execute análise com código atualizado:

```powershell
# Force re-análise
python -m gridsearch.core --analyze-only
```

## 📚 Referências

- **CodeCarbon**: <https://codecarbon.io/>
- **Green AI**: <https://arxiv.org/abs/1907.10597>
- **Energy Costs NLP**: <https://arxiv.org/abs/1906.02243>
- **Sustainable AI**: <https://www.nature.com/articles/s42256-020-0219-9>

---

