# 🚀 AI Web Scraper with OCR, Data Cleaning & RAG System

Um sistema completo de web scraping com OCR, limpeza de dados e sistema RAG (Retrieval-Augmented Generation) para extrair, processar e consultar informações de sites corporativos.

## 📋 Funcionalidades

- **🕷️ Web Scraping Avançado**: Screenshots de alta qualidade com Playwright
- **🔍 OCR Inteligente**: Extração de texto com Tesseract e melhoria de imagens
- **🧹 Limpeza de Dados**: Processamento automático com NLTK e análise linguística
- **🤖 Sistema RAG**: Busca semântica com FAISS e OpenAI
- **🌐 Interface Web**: Dashboard interativo com Streamlit
- **🔐 Autenticação**: Login automático para sites corporativos
- **🌐 Suporte a Proxy**: Configuração automática de proxy corporativo

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Playwright** - Automação de navegador
- **Tesseract OCR** - Extração de texto de imagens
- **OpenAI API** - Embeddings e geração de respostas
- **FAISS** - Busca vetorial de similaridade
- **Streamlit** - Interface web interativa
- **NLTK** - Processamento de linguagem natural
- **Pandas** - Análise de dados

## 📦 Instalação

### 1️⃣ Pré-requisitos

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv tesseract-ocr tesseract-ocr-por

# Windows (usando Chocolatey)
choco install tesseract

# macOS (usando Homebrew)
brew install tesseract tesseract-lang
```

### 2️⃣ Configuração do Ambiente

```bash
# Clone o projeto
git clone <seu-repositorio>
cd DevProjects

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# Instalar navegadores do Playwright
playwright install
```

### 3️⃣ Configuração de Credenciais

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar com suas credenciais
nano .env
```

**Conteúdo do arquivo .env:**

```bash
# OpenAI Configuration
OPENAI_API_KEY=sua_chave_openai_aqui

# Login Credentials
LOGIN_EMAIL=seu.email@empresa.com
LOGIN_PASSWORD=sua_senha_aqui

# Proxy Configuration
USE_PROXY=true
PROXY_PARTNERS_SERVER=http://proxypartners.intratest.com:8080
PROXY_PARTNERS_USERNAME=seu_usuario
PROXY_PARTNERS_PASSWORD=sua_senha_proxy

PROXY_USERS_SERVER=http://proxyusers.intratest.com:8080
PROXY_USERS_USERNAME=seu_usuario
PROXY_USERS_PASSWORD=sua_senha_proxy

# Default Settings
DEFAULT_URL=https://seu-site-corporativo.com
DEFAULT_MAX_DEPTH=2
DEFAULT_CLEAN_DATA=true
DEFAULT_BUILD_RAG=true
DEFAULT_LAUNCH_WEB=true

# Web Interface
WEB_PORT=8501
WEB_HOST=localhost
```

## 🚀 Como Executar

### ⚡ Execução Rápida (Recomendado)

```bash
python run_scraper_with_cleaning.py
```

Este comando executa todo o pipeline automaticamente:
1. ✅ Verifica dependências
2. ✅ Executa web scraping com OCR
3. ✅ Limpa e processa os dados
4. ✅ Constrói sistema RAG
5. ✅ Lança interface web interativa

### 📋 Ordem de Execução Detalhada

#### 1️⃣ **PREPARAÇÃO (Primeira Vez)**

```bash
# Verificar instalação
python run_scraper_with_cleaning.py
# Escolha "y" para mostrar guia de execução

# Verificar CUDA/GPU (opcional)
# Escolha "y" quando perguntado sobre CUDA
```

#### 2️⃣ **PIPELINE COMPLETO**

```bash
python run_scraper_with_cleaning.py
```

**Fluxo interativo:**
- 📋 Mostrar guia de execução? (y/N)
- 🔍 Check CUDA/GPU availability? (y/N)
- 🌐 Enter initial URL: `https://seu-site.com`
- 📊 Enter maximum depth: `2`
- 🌐 Use corporate proxy? (Y/n)
- 🧹 Clean data after scraping? (Y/n)
- 🤖 Build RAG system? (Y/n)
- 🌐 Launch web interface? (Y/n)

#### 3️⃣ **EXECUÇÃO MODULAR** (Opcional)

```bash
# Apenas Web Scraping
python scraping_crawl4ai.py

# Apenas Limpeza de Dados
python data_cleaner.py

# Apenas Sistema RAG
python rag_system.py

# Apenas Interface Web
streamlit run rag_web_interface.py
```

## 📂 Estrutura de Arquivos

```
DevProjects/
├── 📄 README.md                     # Documentação
├── 📄 requirements.txt              # Dependências Python
├── 📄 .env                         # Configurações (criar)
├── 📄 .env.example                 # Exemplo de configurações
│
├── 🐍 run_scraper_with_cleaning.py # Script principal
├── 🐍 scraping_crawl4ai.py         # Módulo de scraping
├── 🐍 data_cleaner.py              # Módulo de limpeza
├── 🐍 rag_system.py                # Sistema RAG
├── 🐍 rag_web_interface.py         # Interface web
│
└── 📁 scraped_data/                # Dados gerados
    ├── 📁 screenshots/             # Screenshots originais
    ├── 📁 enhanced/                # Imagens melhoradas
    ├── 📁 texts/                   # Textos extraídos
    ├── 📁 cleaned/                 # Dados limpos
    ├── 📁 analytics/               # Relatórios
    └── 📁 rag_index/               # Índice vetorial
```

## 🔧 Verificação do Sistema

### Testar Dependências

```bash
python -c "
import playwright, tesseract, faiss, openai, streamlit
print('✅ Todas as dependências instaladas!')
"
```

### Testar CUDA (GPU)

```bash
python -c "
import torch
print(f'CUDA disponível: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

### Testar OCR

```bash
python -c "
import pytesseract
print('✅ Tesseract funcionando!')
print(f'Versão: {pytesseract.get_tesseract_version()}')
"
```

## 📊 Resultados Esperados

Após a execução completa, você terá:

### 📁 **scraped_data/texts/**
- Textos extraídos com metadados
- Informações de proxy, tempo, confiança OCR

### 📁 **scraped_data/cleaned/**
- Dados limpos em JSON e TXT
- Remoção de ruído e artefatos OCR
- Análise linguística e palavras-chave

### 📁 **scraped_data/analytics/**
- Relatórios de limpeza (CSV/JSON)
- Estatísticas de eficiência
- Distribuição de idiomas

### 📁 **scraped_data/rag_index/**
- Índice vetorial FAISS
- Embeddings OpenAI
- Sistema de busca semântica

### 🌐 **Interface Web** (http://localhost:8501)
- Chat interativo com dados
- Busca por similaridade
- Visualização de fontes

## 🔍 Exemplos de Uso

### Consulta via Interface Web

1. Acesse: http://localhost:8501
2. Digite: "Quais são os principais produtos da empresa?"
3. Obtenha resposta baseada nos dados coletados

### Consulta via Código

```python
from rag_system import RAGSystem

rag = RAGSystem("scraped_data")
result = await rag.query("informações sobre vendas")

print(f"Resposta: {result['answer']}")
print(f"Confiança: {result['confidence']}")
print(f"Fontes: {result['sources']}")
```

## 🐛 Solução de Problemas

### Erro de Dependências

```bash
pip install -r requirements.txt
playwright install
```

### Erro de Tesseract

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-por

# Verificar instalação
tesseract --version
```

### Erro de Proxy

```bash
# Testar conectividade
curl -x http://usuario:senha@proxy:8080 http://httpbin.org/ip

# Verificar credenciais no .env
```

### Erro OpenAI API

```bash
# Verificar chave no .env
echo $OPENAI_API_KEY

# Testar API
python -c "
import openai
client = openai.OpenAI()
print('✅ OpenAI API funcionando!')
"
```

## 📝 Logs e Debugging

### Logs Detalhados

```bash
python run_scraper_with_cleaning.py 2>&1 | tee execution.log
```

### Verificar Arquivos Gerados

```bash
# Contar arquivos processados
find scraped_data -name "*.txt" | wc -l
find scraped_data -name "*_cleaned.json" | wc -l

# Ver último arquivo processado
ls -la scraped_data/texts/ | tail -5
```

## 🔒 Segurança

- ✅ Credenciais em arquivo .env (não versionado)
- ✅ Suporte a proxy corporativo
- ✅ Headers realistas para evitar detecção
- ✅ Controle de rate limiting
- ✅ Logs sem credenciais sensíveis

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🆘 Suporte

- 📧 Email: seu.email@empresa.com
- 📱 Teams: @seu_usuario
- 🐛 Issues: [GitHub Issues](https://github.com/seu-usuario/projeto/issues)

---

**⚡ Início Rápido:**

```bash
pip install -r requirements.txt
playwright install
cp .env.example .env
# Editar .env com suas credenciais
python run_scraper_with_cleaning.py
```

🎉 **Pronto! Seu sistema está funcionando!**
