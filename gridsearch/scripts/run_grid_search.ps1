# Grid Search PowerShell Launcher
# BERT-PLI Project

param(
    [Parameter(Position=0)]
    [ValidateSet("test", "full", "resume", "analyze")]
    [string]$Mode = "test",
    
    [int]$Parallel = 2,
    
    [string]$Config = "config/experiments/BertPLI.config",
    
    [string]$SearchConfig = ""
)

# Configurações
$ErrorActionPreference = "Stop"
$PROJECT_ROOT = $PSScriptRoot | Split-Path | Split-Path

# Cores
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# Banner
Write-ColorOutput "`n============================================" "Cyan"
Write-ColorOutput "     BERT-PLI Grid Search Launcher" "Cyan"
Write-ColorOutput "============================================`n" "Cyan"

# Determina arquivo de configuração
if ($SearchConfig -eq "") {
    switch ($Mode) {
        "test" {
            $SearchConfig = "gridsearch/config/grid_search_test.json"
            Write-ColorOutput "Modo: TESTE (8 experimentos)" "Yellow"
        }
        "full" {
            $SearchConfig = "gridsearch/config/grid_search.json"
            Write-ColorOutput "Modo: COMPLETO (216 experimentos)" "Yellow"
        }
        "resume" {
            Write-ColorOutput "Modo: RETOMAR execução anterior" "Yellow"
        }
        "analyze" {
            Write-ColorOutput "Modo: ANÁLISE de resultados" "Yellow"
        }
    }
}

Write-ColorOutput "Configuração base: $Config" "Gray"
if ($SearchConfig -ne "") {
    Write-ColorOutput "Grid config: $SearchConfig" "Gray"
}
Write-ColorOutput "Workers paralelos: $Parallel`n" "Gray"

# Validações
if ($Mode -ne "analyze" -and $Mode -ne "resume") {
    if (-not (Test-Path $Config)) {
        Write-ColorOutput "ERRO: Arquivo de configuração não encontrado: $Config" "Red"
        exit 1
    }
    
    if (-not (Test-Path $SearchConfig)) {
        Write-ColorOutput "ERRO: Arquivo de grid search não encontrado: $SearchConfig" "Red"
        exit 1
    }
}

# Monta comando
$Command = "python -m gridsearch.core"

switch ($Mode) {
    "analyze" {
        $Command += " --analyze-only"
    }
    "resume" {
        $Command += " --resume --parallel $Parallel"
    }
    default {
        $Command += " --config `"$Config`" --search-config `"$SearchConfig`" --parallel $Parallel"
    }
}

# Confirmação
Write-ColorOutput "Comando a executar:" "Cyan"
Write-ColorOutput "  $Command`n" "White"

if ($Mode -eq "full") {
    Write-ColorOutput "ATENÇÃO: Execução completa pode levar várias horas!" "Yellow"
    $confirm = Read-Host "Deseja continuar? (s/N)"
    if ($confirm -ne "s" -and $confirm -ne "S") {
        Write-ColorOutput "Execução cancelada pelo usuário." "Yellow"
        exit 0
    }
}

# Executa
Write-ColorOutput "`nIniciando Grid Search...`n" "Green"

try {
    Invoke-Expression $Command
    
    Write-ColorOutput "`n============================================" "Green"
    Write-ColorOutput "     Grid Search concluído com sucesso!" "Green"
    Write-ColorOutput "============================================`n" "Green"
    
} catch {
    Write-ColorOutput "`n============================================" "Red"
    Write-ColorOutput "     ERRO durante execução do Grid Search" "Red"
    Write-ColorOutput "============================================" "Red"
    Write-ColorOutput $_.Exception.Message "Red"
    exit 1
}

# Resultados
$ResultsFile = "output/experiments/grid_search/grid_search_results.json"
$SummaryFile = "output/experiments/grid_search/grid_search_summary.txt"

if (Test-Path $SummaryFile) {
    Write-ColorOutput "`nRESUMO DOS RESULTADOS:" "Cyan"
    Get-Content $SummaryFile | Write-Host
}

Write-ColorOutput "`nArquivos gerados:" "Cyan"
if (Test-Path $ResultsFile) {
    Write-ColorOutput "  - $ResultsFile" "Gray"
}
if (Test-Path $SummaryFile) {
    Write-ColorOutput "  - $SummaryFile" "Gray"
}
Write-Host ""
