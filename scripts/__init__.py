"""
Scripts CLI — BERT-PLI
======================
Entry points para execução do pipeline via linha de comando.

Cada módulo expõe uma função main() registrada em pyproject.toml
como console_script, permitindo tanto:

    bert-pli-train --config ...          (após uv pip install -e .)
    python -m scripts.train --config ... (direto)
"""
