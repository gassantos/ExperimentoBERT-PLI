# Grid Search - Guia Rápido (5 minutos)

## 🎯 Objetivo

Executar busca em grade de hiperparâmetros para encontrar a melhor configuração do BERT-PLI.

## 🚀 Inicio Rápido

### Passo 1: Teste Inicial (8 experimentos, ~2-3 horas)

```powershell
# Via PowerShell
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 2

# Ou via Python
python -m gridsearch.core --config config/experiments/BertPLI.config \
                          --search-config gridsearch/config/grid_search_test.json \
                          --parallel 2
```

### Passo 2: Aguarde e Monitore

O script mostra o progresso em tempo real:
```
[INFO] Total de experimentos: 8
[INFO] Progresso: 1/8
[INFO] Progresso: 2/8
...
[INFO] Progresso: 8/8
```

### Passo 3: Veja os Resultados

Ao final, um resumo é exibido automaticamente:

```
===============================================================================
GRID SEARCH - RELATÓRIO DE RESULTADOS
===============================================================================
Data: 15/02/2026 14:30:00

RESUMO GERAL:
  Total de experimentos: 8
  Bem-sucedidos: 8
  Falhos: 0

MELHOR CONFIGURAÇÃO (Tempo de Treinamento):
  Experimento: 3
  Tempo: 1234.56 segundos
  Parâmetros:
    - learning_rate: 2e-5
    - batch_size: 16
    - optimizer: adam
===============================================================================
```

### Passo 4: Análise Detalhada (Opcional)

```powershell
# Análise estatística completa
python -m gridsearch.analysis --results-file output/experiments/grid_search/grid_search_results.json
```

## 📊 Interpretação dos Resultados

### Arquivos Gerados

1. **grid_search_results.json** - Dados brutos completos
2. **grid_search_summary.txt** - Resumo legível
3. **analysis/** - Análises estatísticas detalhadas

### Métricas Principais

- **tempo de treinamento** - Qual configuração é mais rápida
- **energia consumida** - Qual configuração é mais eficiente
- **uso de RAM** - Qual configuração usa menos memória

## ⚡ Próximos Passos

### Se os resultados foram bons:
```powershell
# Execute o grid completo (216 experimentos)
.\gridsearch\scripts\run_grid_search.ps1 -Mode full -Parallel 2
```

### Se houve problemas de memória:
```powershell
# Reduza o paralelismo
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 1
```

### Se a execução foi interrompida:
```powershell
# Retome de onde parou
.\gridsearch\scripts\run_grid_search.ps1 -Mode resume -Parallel 2
```

## 🛡️ Requisitos de Sistema

### Para Grid de Teste (8 experimentos)
- **RAM:** 16 GB mínimo
- **Tempo:** ~2-3 horas
- **Espaço em disco:** ~5 GB

### Para Grid Completo (216 experimentos)
- **RAM:** 32 GB recomendado
- **Tempo:** ~72-108 horas (3-4 dias)
- **Espaço em disco:** ~50 GB

## 🆘 Problemas Comuns

### "Out of Memory"
```powershell
# Solução: Reduza workers paralelos
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 1
```

### Processo travado
```
Ctrl+C para interromper
Depois execute:
.\gridsearch\scripts\run_grid_search.ps1 -Mode resume
```

### Resultados estranhos
```powershell
# Limpe e recomece
Remove-Item -Recurse output/experiments/grid_search/*
.\gridsearch\scripts\run_grid_search.ps1 -Mode test -Parallel 2
```

## 📚 Mais Informações

- **Guia Completo:** `gridsearch/docs/GUIDE.md`
- **Visão Técnica:** `gridsearch/docs/OVERVIEW.md`
- **README:** `gridsearch/README.md`

## ✅ Checklist

- [ ] Executei o teste com 8 experimentos
- [ ] Verifiquei os resultados
- [ ] Configurei o paralelismo adequado para meu sistema
- [ ] Li o resumo gerado
- [ ] Estou pronto para o grid completo

---

**Tempo total estimado deste tutorial:** 5 minutos + 2-3 horas de execução

**Próximo passo:** Executar grid completo ou personalizar hiperparâmetros
