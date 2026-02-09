FROM python:3.7-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Dependências de sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cria ambiente virtual explícito
RUN python -m venv /opt/venv

# Instala uv (binário oficial)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Garante que uv está no PATH
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace

# Copia requirements
COPY requirements.txt .

# Instala PyTorch CPU compatível com BERT legado
RUN uv pip install \
    torch==1.2.0 \
    torchvision==0.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Instala dependências do projeto
RUN uv pip install -r requirements.txt

# Ferramentas científicas auxiliares
RUN uv pip install \
    jupyter \
    ipykernel \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn

# Kernel Jupyter nomeado
RUN python -m ipykernel install \
    --name bert-pli \
    --display-name "Python (BERT-PLI)"

CMD ["bash"]
