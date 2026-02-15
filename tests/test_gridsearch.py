"""
Script de teste para o módulo gridsearch
Testa todas as funcionalidades principais antes do commit
"""

import sys
import json
from pathlib import Path

print("=" * 70)
print("TESTES DO MÓDULO GRIDSEARCH")
print("=" * 70)

# Teste 1: Importação
print("\n[1/6] Teste de Importação...")
try:
    from gridsearch import (
        run_grid_search,
        generate_parameter_grid,
        analyze_results
    )
    from gridsearch.utils import (
        check_memory_availability,
        estimate_memory_requirements,
        filter_grid_config
    )
    from gridsearch.analysis import (
        compute_descriptive_statistics,
        analyze_correlations,
        rank_configurations
    )
    print("✓ Todas as importações bem-sucedidas")
except Exception as e:
    print(f"✗ Erro na importação: {e}")
    sys.exit(1)

# Teste 2: Geração de combinações
print("\n[2/6] Teste de Geração de Combinações...")
try:
    test_grid = {
        "learning_rate": [1e-5, 2e-5],
        "batch_size": [8, 16]
    }
    
    combinations = generate_parameter_grid(test_grid)
    
    expected_count = 4  # 2 x 2
    actual_count = len(combinations)
    
    if actual_count == expected_count:
        print(f"✓ Geradas {actual_count} combinações (esperado: {expected_count})")
        print(f"  Exemplo: {combinations[0]}")
    else:
        print(f"✗ Erro: {actual_count} combinações (esperado: {expected_count})")
        sys.exit(1)
except Exception as e:
    print(f"✗ Erro na geração: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 3: Validação de Memória
print("\n[3/6] Teste de Validação de Memória...")
try:
    # Teste com configuração segura
    is_safe, message = check_memory_availability(
        parallel=2,
        batch_size=16
    )
    
    print(f"✓ Validação executada")
    print(f"  Resultado: {'SEGURO' if is_safe else 'INSEGURO'}")
    print(f"  Mensagem:\n{message}")
    
    # Teste de estimativa
    estimated_gb = estimate_memory_requirements(
        parallel=2,
        batch_size=16
    )
    
    print(f"✓ Estimativa: {estimated_gb:.2f} GB necessários")
    
except Exception as e:
    print(f"✗ Erro na validação: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 4: Filtragem de Configuração
print("\n[4/6] Teste de Filtragem de Grid Config...")
try:
    config_with_metadata = {
        "description": "Test grid",
        "notes": ["test note"],
        "learning_rate": [1e-5],
        "batch_size": [16]
    }
    
    filtered = filter_grid_config(config_with_metadata)
    
    if "description" not in filtered and "learning_rate" in filtered:
        print(f"✓ Filtragem funcionando")
        print(f"  Parâmetros filtrados: {list(filtered.keys())}")
    else:
        print(f"✗ Erro na filtragem: {filtered}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Erro na filtragem: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 5: Carregamento de Configurações JSON
print("\n[5/6] Teste de Carregamento de Configs JSON...")
try:
    # Testa grid de teste
    test_config_file = Path("gridsearch/config/grid_search_test.json")
    
    if test_config_file.exists():
        with open(test_config_file) as f:
            test_config = json.load(f)
        
        if "hyperparameters" in test_config:
            params = test_config["hyperparameters"]
            combos = generate_parameter_grid(params)
            print(f"✓ Grid de teste carregado: {len(combos)} combinações")
        else:
            print(f"✗ Estrutura inválida no JSON")
            sys.exit(1)
    else:
        print(f"✗ Arquivo não encontrado: {test_config_file}")
        sys.exit(1)
    
    # Testa grid completo
    full_config_file = Path("gridsearch/config/grid_search.json")
    
    if full_config_file.exists():
        with open(full_config_file) as f:
            full_config = json.load(f)
        
        if "hyperparameters" in full_config:
            params = full_config["hyperparameters"]
            combos = generate_parameter_grid(params)
            print(f"✓ Grid completo carregado: {len(combos)} combinações")
        else:
            print(f"✗ Estrutura inválida no JSON")
            sys.exit(1)
    else:
        print(f"✗ Arquivo não encontrado: {full_config_file}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Erro no carregamento: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Teste 6: Análise de Resultados Mock
print("\n[6/6] Teste de Análise com Dados Mock...")
try:
    # Cria resultados simulados
    mock_results = [
        {
            "grid_experiment_idx": 0,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 8},
            "status": "success",
            "resources": {
                "train_time_sec": 1234.5,
                "energy_kwh": 0.025,
                "peak_ram_mb": 8192.0
            }
        },
        {
            "grid_experiment_idx": 1,
            "grid_params": {"learning_rate": 2e-5, "batch_size": 16},
            "status": "success",
            "resources": {
                "train_time_sec": 1156.3,
                "energy_kwh": 0.023,
                "peak_ram_mb": 12288.0
            }
        },
        {
            "grid_experiment_idx": 2,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 16},
            "status": "failed",
            "error": "Out of memory"
        }
    ]
    
    # Testa estatísticas descritivas
    stats = compute_descriptive_statistics(mock_results)
    
    if stats and "train_time" in stats:
        print(f"✓ Estatísticas computadas")
        print(f"  Tempo médio: {stats['train_time']['mean']:.2f}s")
        print(f"  Experimentos bem-sucedidos: {stats['successful_experiments']}")
        print(f"  Experimentos falhos: {stats['failed_experiments']}")
    else:
        print(f"✗ Erro nas estatísticas: {stats}")
        sys.exit(1)
    
    # Testa ranking
    ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
    
    if ranked and len(ranked) > 0:
        print(f"✓ Ranking gerado: {len(ranked)} configurações")
        print(f"  Melhor: {ranked[0]['params']}")
    else:
        print(f"✗ Erro no ranking")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Erro na análise: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Resumo final
print("\n" + "=" * 70)
print("RESUMO DOS TESTES")
print("=" * 70)
print("✓ [1/6] Importações")
print("✓ [2/6] Geração de combinações")
print("✓ [3/6] Validação de memória")
print("✓ [4/6] Filtragem de configs")
print("✓ [5/6] Carregamento de JSONs")
print("✓ [6/6] Análise de resultados")
print("\n🎉 TODOS OS TESTES PASSARAM!")
print("=" * 70)

sys.exit(0)
