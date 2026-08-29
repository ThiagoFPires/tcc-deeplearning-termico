# Imagem base oficial do Python Slim
FROM python:3.11-slim

# Metadados do autor
LABEL maintainer="Thiago Freitas Pires <contato.thiagofreitasp@gmail.com>"
LABEL description="ThermoScan AI - Classificador Térmico Mamário com Deep Learning"

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Define diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias para OpenCV/PIL e compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências do Python (PyTorch CPU para portabilidade em qualquer servidor)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    scikit-learn \
    pillow \
    matplotlib \
    numpy \
    pandas

# Copia código-fonte do projeto
COPY api/ ./api/
COPY interface/ ./interface/
COPY treinamento/ ./treinamento/
COPY modelos_salvos/ ./modelos_salvos/

# Expõe a porta 8000 (FastAPI + Front-end)
EXPOSE 8000

# Inicializa o servidor FastAPI com Uvicorn
CMD ["uvicorn", "main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]
