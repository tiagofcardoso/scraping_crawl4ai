# 🚀 AI Web Scraper with OCR, Data Cleaning & RAG System

A complete web scraping system with OCR, data cleaning and RAG (Retrieval-Augmented Generation) for extracting, processing and querying information from corporate websites.

## 📋 Features

- **🕷️ Advanced Web Scraping**: High-quality screenshots with Playwright
- **🔍 Intelligent OCR**: Text extraction with Tesseract and image enhancement
- **🧹 Data Cleaning**: Automatic processing with NLTK and linguistic analysis
- **🤖 RAG System**: Semantic search with FAISS and OpenAI
- **🌐 Web Interface**: Interactive dashboard with Streamlit
- **🔐 Authentication**: Automatic login for corporate sites
- **🌐 Proxy Support**: Automatic corporate proxy configuration

## 🛠️ Technologies Used

- **Python 3.8+**
- **Playwright** - Browser automation
- **Tesseract OCR** - Text extraction from images
- **OpenAI API** - Embeddings and response generation
- **FAISS** - Vector similarity search
- **Streamlit** - Interactive web interface
- **NLTK** - Natural language processing
- **Pandas** - Data analysis

## 📦 Installation

### 1️⃣ Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv tesseract-ocr tesseract-ocr-por

# Windows (using Chocolatey)
choco install tesseract

# macOS (using Homebrew)
brew install tesseract tesseract-lang
```

### 2️⃣ Environment Setup

```bash
# Clone the project
git clone <your-repository>
cd DevProjects

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### 3️⃣ Credentials Configuration

```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
nano .env
```

**.env file content:**

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key_here

# Login Credentials
LOGIN_EMAIL=your.email@company.com
LOGIN_PASSWORD=your_password_here

# Proxy Configuration
USE_PROXY=true
PROXY_PARTNERS_SERVER=http://proxypartners.intratest.com:8080
PROXY_PARTNERS_USERNAME=your_username
PROXY_PARTNERS_PASSWORD=your_proxy_password

PROXY_USERS_SERVER=http://proxyusers.intratest.com:8080
PROXY_USERS_USERNAME=your_username
PROXY_USERS_PASSWORD=your_proxy_password

# Default Settings
DEFAULT_URL=https://your-corporate-site.com
DEFAULT_MAX_DEPTH=2
DEFAULT_CLEAN_DATA=true
DEFAULT_BUILD_RAG=true
DEFAULT_LAUNCH_WEB=true

# Web Interface
WEB_PORT=8501
WEB_HOST=localhost
```

## 🚀 How to Run

### ⚡ Quick Start (Recommended)

```bash
python run_scraper_with_cleaning.py
```

This command runs the entire pipeline automatically:
1. ✅ Checks dependencies
2. ✅ Executes web scraping with OCR
3. ✅ Cleans and processes data
4. ✅ Builds RAG system
5. ✅ Launches interactive web interface

### 📋 Detailed Execution Order

#### 1️⃣ **PREPARATION (First Time)**

```bash
# Check installation
python run_scraper_with_cleaning.py
# Choose "y" to show execution guide

# Check CUDA/GPU (optional)
# Choose "y" when asked about CUDA
```

#### 2️⃣ **COMPLETE PIPELINE**

```bash
python run_scraper_with_cleaning.py
```

**Interactive flow:**
- 📋 Show execution guide? (y/N)
- 🔍 Check CUDA/GPU availability? (y/N)
- 🌐 Enter initial URL: `https://your-site.com`
- 📊 Enter maximum depth: `2`
- 🌐 Use corporate proxy? (Y/n)
- 🧹 Clean data after scraping? (Y/n)
- 🤖 Build RAG system? (Y/n)
- 🌐 Launch web interface? (Y/n)

#### 3️⃣ **MODULAR EXECUTION** (Optional)

```bash
# Web Scraping Only
python scraping_crawl4ai.py

# Data Cleaning Only
python data_cleaner.py

# RAG System Only
python rag_system.py

# Web Interface Only
streamlit run rag_web_interface.py
```

## 📂 File Structure

```
DevProjects/
├── 📄 README.md                     # Documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env                         # Configuration (create)
├── 📄 .env.example                 # Configuration example
│
├── 🐍 run_scraper_with_cleaning.py # Main script
├── 🐍 scraping_crawl4ai.py         # Scraping module
├── 🐍 data_cleaner.py              # Cleaning module
├── 🐍 rag_system.py                # RAG system
├── 🐍 rag_web_interface.py         # Web interface
│
└── 📁 scraped_data/                # Generated data
    ├── 📁 screenshots/             # Original screenshots
    ├── 📁 enhanced/                # Enhanced images
    ├── 📁 texts/                   # Extracted texts
    ├── 📁 cleaned/                 # Cleaned data
    ├── 📁 analytics/               # Reports
    └── 📁 rag_index/               # Vector index
```

## 🔧 System Verification

### Test Dependencies

```bash
python -c "
import playwright, tesseract, faiss, openai, streamlit
print('✅ All dependencies installed!')
"
```

### Test CUDA (GPU)

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

### Test OCR

```bash
python -c "
import pytesseract
print('✅ Tesseract working!')
print(f'Version: {pytesseract.get_tesseract_version()}')
"
```

## 📊 Expected Results

After complete execution, you will have:

### 📁 **scraped_data/texts/**
- Extracted texts with metadata
- Proxy information, time, OCR confidence

### 📁 **scraped_data/cleaned/**
- Clean data in JSON and TXT
- OCR artifacts and noise removal
- Linguistic analysis and keywords

### 📁 **scraped_data/analytics/**
- Cleaning reports (CSV/JSON)
- Efficiency statistics
- Language distribution

### 📁 **scraped_data/rag_index/**
- FAISS vector index
- OpenAI embeddings
- Semantic search system

### 🌐 **Web Interface** (http://localhost:8501)
- Interactive chat with data
- Similarity search
- Source visualization

## 🔍 Usage Examples

### Query via Web Interface

1. Access: http://localhost:8501
2. Type: "What are the company's main products?"
3. Get answer based on collected data

### Query via Code

```python
from rag_system import RAGSystem

rag = RAGSystem("scraped_data")
result = await rag.query("sales information")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

## 🐛 Troubleshooting

### Dependencies Error

```bash
pip install -r requirements.txt
playwright install
```

### Tesseract Error

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-por

# Check installation
tesseract --version
```

### Proxy Error

```bash
# Test connectivity
curl -x http://user:password@proxy:8080 http://httpbin.org/ip

# Check credentials in .env
```

### OpenAI API Error

```bash
# Check key in .env
echo $OPENAI_API_KEY

# Test API
python -c "
import openai
client = openai.OpenAI()
print('✅ OpenAI API working!')
"
```

## 📝 Logs and Debugging

### Detailed Logs

```bash
python run_scraper_with_cleaning.py 2>&1 | tee execution.log
```

### Check Generated Files

```bash
# Count processed files
find scraped_data -name "*.txt" | wc -l
find scraped_data -name "*_cleaned.json" | wc -l

# View last processed file
ls -la scraped_data/texts/ | tail -5
```

## 🔒 Security

- ✅ Credentials in .env file (not versioned)
- ✅ Corporate proxy support
- ✅ Realistic headers to avoid detection
- ✅ Rate limiting control
- ✅ Logs without sensitive credentials

## 🤝 Contributing

1. Fork the project
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT license. See the `LICENSE` file for details.

## 🆘 Support

- 📧 Email: your.email@company.com
- 📱 Teams: @your_username
- 🐛 Issues: [GitHub Issues](https://github.com/your-user/project/issues)

---

**⚡ Quick Start:**

```bash
pip install -r requirements.txt
playwright install
cp .env.example .env
# Edit .env with your credentials
python run_scraper_with_cleaning.py
```

🎉 **Ready! Your system is working!**
