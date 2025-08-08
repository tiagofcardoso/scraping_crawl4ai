import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_cuda_availability():
    """Check CUDA availability and GPU information"""
    print("\n🔍 Checking CUDA and GPU availability...")
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available!")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU memory allocation
            try:
                device = torch.device('cuda')
                test_tensor = torch.randn(1000, 1000, device=device)
                print(f"✅ GPU memory allocation test: SUCCESS")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU memory allocation test failed: {e}")
        else:
            print(f"❌ CUDA not available in PyTorch")
            print(f"   Reasons might be:")
            print(f"   - CUDA not installed")
            print(f"   - PyTorch CPU-only version")
            print(f"   - GPU drivers not installed")
    except ImportError:
        print(f"⚠️  PyTorch not installed")
    
    # Check TensorFlow CUDA (if available)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU devices: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print(f"❌ No GPU devices found by TensorFlow")
    except ImportError:
        print(f"⚠️  TensorFlow not installed")
    
    # Check NVIDIA System Management Interface
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ nvidia-smi available:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA-SMI' in line or 'Driver Version' in line or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
                elif 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'Quadro' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"❌ nvidia-smi not available or failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ nvidia-smi command not found")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
    
    # Check FAISS GPU support
    try:
        import faiss
        print(f"✅ FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'}")
        
        # Check if FAISS GPU is available
        try:
            # Try to get GPU resources
            ngpus = faiss.get_num_gpus()
            print(f"✅ FAISS GPU support: {ngpus} GPU(s) available")
            
            if ngpus > 0:
                for i in range(ngpus):
                    res = faiss.StandardGpuResources()
                    print(f"   GPU {i}: Available for FAISS")
        except Exception as e:
            print(f"⚠️  FAISS GPU support: Not available ({e})")
            print(f"   Using CPU version of FAISS")
    except ImportError:
        print(f"⚠️  FAISS not installed")
    
    print("\n" + "="*60)

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import faiss
        print(f"✅ FAISS available: {faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'}")
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers available")
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import streamlit
        print(f"✅ Streamlit available")
    except ImportError:
        missing_deps.append("streamlit")
    
    # Check for additional ML dependencies
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        # Check if PyTorch has GPU support
        if torch.cuda.is_available():
            print(f"🎯 GPU acceleration available with your RTX 3050!")
            print(f"   Note: faiss-gpu may not be available via pip for all systems")
            gpu_available = True
        else:
            gpu_available = False
            
    except ImportError:
        print(f"⚠️  PyTorch not available - some features may be limited")
        missing_deps.append("torch")
        gpu_available = False
    
    if missing_deps:
        print("❌ Missing dependencies detected:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n🔧 To install missing dependencies, run:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("\n💡 Or install all requirements:")
        print("   pip install -r requirements.txt")
        
        if gpu_available:
            print("\n🚀 For GPU acceleration (RTX 3050 detected):")
            print("   # Option 1: Try conda for faiss-gpu")
            print("   conda install -c conda-forge faiss-gpu")
            print("   # Option 2: Use faiss-cpu (still fast)")
            print("   pip install faiss-cpu")
            print("   # Note: faiss-cpu will work fine for most workloads!")
        else:
            print("\n💻 Installing CPU versions:")
            print("   pip install faiss-cpu")
        
        return False
    
    return True

def get_env_bool(key, default=False):
    """Convert environment variable to boolean"""
    value = os.getenv(key, str(default)).lower()
    return value in ['true', '1', 'yes', 'on']

def show_execution_guide():
    """Show step-by-step execution guide"""
    print("📋 GUIA DE EXECUÇÃO DO PROJETO")
    print("=" * 60)
    print("1️⃣  PREPARAR AMBIENTE:")
    print("   • Criar arquivo .env com suas credenciais")
    print("   • pip install -r requirements.txt")
    print("   • playwright install")
    print()
    print("2️⃣  EXECUTAR PIPELINE COMPLETO:")
    print("   • python run_scraper_with_cleaning.py")
    print("   • Segue: Scraper → Cleaner → RAG → Web Interface")
    print()
    print("3️⃣  OU EXECUTAR MÓDULOS INDIVIDUAIS:")
    print("   • python scraping_crawl4ai.py (apenas scraping)")
    print("   • python data_cleaner.py (apenas limpeza)")
    print("   • python rag_system.py (apenas RAG)")
    print()
    print("4️⃣  VERIFICAR RESULTADOS:")
    print("   • scraped_data/texts/ (textos extraídos)")
    print("   • scraped_data/cleaned/ (dados limpos)")
    print("   • scraped_data/analytics/ (relatórios)")
    print("=" * 60)

async def main():
    """Run complete workflow: Scraper -> Cleaner -> RAG"""
    print("🚀 Complete AI Pipeline: Scraper + Cleaner + RAG")
    print("=" * 60)
    
    # Show execution guide option
    guide_check = input("Show execution guide? (y/N): ").strip().lower()
    if guide_check in ['y', 'yes', 's', 'sim']:
        show_execution_guide()
        if input("\nContinue with execution? (Y/n): ").strip().lower() in ['n', 'no', 'não']:
            return
    
    # Check .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found!")
        print("🔧 Create a .env file with your credentials:")
        print("   • OPENAI_API_KEY=your_key_here")
        print("   • LOGIN_EMAIL=your_email@company.com")
        print("   • LOGIN_PASSWORD=your_password")
        print("   • PROXY_PARTNERS_USERNAME=your_username")
        print("   • PROXY_PARTNERS_PASSWORD=your_proxy_password")
        print()
        if input("Continue anyway? (y/N): ").strip().lower() not in ['y', 'yes']:
            return
    
    # Add option to check CUDA
    cuda_check = input("Check CUDA/GPU availability? (y/N): ").strip().lower()
    if cuda_check in ['y', 'yes']:
        check_cuda_availability()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        print("\n💡 Quick install commands:")
        print("   pip install sentence-transformers streamlit faiss-cpu")
        print("   # This will work great with your RTX 3050!")
        print("\n🔧 Alternative with conda (for GPU FAISS):")
        print("   conda install -c conda-forge faiss-gpu sentence-transformers")
        print("   pip install streamlit")
        sys.exit(1)
    
    # Import modules after dependency check
    try:
        from scraping_crawl4ai import ScreenshotScraper
        from data_cleaner import DataCleaner
        from rag_system import RAGSystem
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure all modules are in the same directory and dependencies are installed.")
        sys.exit(1)
    
    # Get inputs with environment variable defaults
    url = input(f"Enter initial URL (default: {os.getenv('DEFAULT_URL', 'https://example.com')}): ").strip()
    if not url:
        url = os.getenv('DEFAULT_URL', 'https://example.com')
        print(f"Using default URL: {url}")
    
    depth = input(f"Enter maximum depth (default: {os.getenv('DEFAULT_MAX_DEPTH', '2')}): ").strip()
    try:
        max_depth = int(depth) if depth else int(os.getenv('DEFAULT_MAX_DEPTH', '2'))
    except ValueError:
        max_depth = int(os.getenv('DEFAULT_MAX_DEPTH', '2'))
    
    # Use environment variables for proxy configuration
    use_proxy_default = get_env_bool('USE_PROXY', False)
    use_proxy_input = input(f"Use corporate proxy? ({'Y' if use_proxy_default else 'n'}/{'n' if use_proxy_default else 'Y'}): ").strip().lower()
    use_proxy = use_proxy_input not in ['n', 'no'] if use_proxy_input else use_proxy_default
    
    clean_data_default = get_env_bool('DEFAULT_CLEAN_DATA', True)
    clean_data_input = input(f"Clean data after scraping? ({'Y' if clean_data_default else 'n'}/{'n' if clean_data_default else 'Y'}): ").strip().lower()
    clean_data = clean_data_input not in ['n', 'no'] if clean_data_input else clean_data_default
    
    build_rag_default = get_env_bool('DEFAULT_BUILD_RAG', True)
    build_rag_input = input(f"Build RAG system? ({'Y' if build_rag_default else 'n'}/{'n' if build_rag_default else 'Y'}): ").strip().lower()
    build_rag = build_rag_input not in ['n', 'no'] if build_rag_input else build_rag_default
    
    launch_web_default = get_env_bool('DEFAULT_LAUNCH_WEB', True)
    launch_web_input = input(f"Launch web interface? ({'Y' if launch_web_default else 'n'}/{'n' if launch_web_default else 'Y'}): ").strip().lower()
    launch_web = launch_web_input not in ['n', 'no'] if launch_web_input else launch_web_default
    
    # PHASE 1: Web Scraping
    print("\n" + "="*60)
    print("PHASE 1: WEB SCRAPING")
    print("="*60)
    
    # Ask about output format in the main pipeline
    print("📋 Choose capture format:")
    print("1. Screenshot only (OCR only)")
    print("2. PDF only (OCR only)")
    print("3. 🚀 Both (Smart OCR comparison) [RECOMMENDED]")
    
    format_choice = input("Choose format (1/2/3, default: 3): ").strip()
    
    if format_choice == "1":
        output_format = "screenshot"
    elif format_choice == "2":
        output_format = "pdf"
    else:  # Default to "both" (option 3)
        output_format = "both"
    
    print(f"✅ Selected format: {output_format}")
    print("ℹ️  Note: Only OCR extraction will be used (no HTML parsing)")
    
    scraper = ScreenshotScraper(use_proxy=use_proxy, output_format=output_format)
    await scraper.run(url, max_depth=max_depth)
    
    # PHASE 2: Data Cleaning
    if clean_data:
        print("\n" + "="*60)
        print("PHASE 2: DATA CLEANING")
        print("="*60)
        
        # Check if there are files to clean
        import glob
        text_files = glob.glob(os.path.join(scraper.output_dir, "texts", "*.txt"))
        
        if text_files:
            print(f"🔍 Found {len(text_files)} text files to clean")
            cleaner = DataCleaner(scraper.output_dir)
            analytics = await cleaner.clean_all_files()
            
            if analytics:
                await cleaner.export_to_csv()
                print(f"📈 Cleaning efficiency: {analytics['summary']['overall_reduction_percent']:.1f}% reduction")
        else:
            print("⚠️  No text files found to clean!")
            print("💡 This might be because:")
            print("   • Screenshot/PDF capture failed")
            print("   • OCR couldn't extract text")
            print("   • Network issues prevented page loading")
            print("   • Try reducing max_depth or using different URL")

    # PHASE 3: RAG System Building
    if build_rag:
        print("\n" + "="*60)
        print("PHASE 3: RAG SYSTEM BUILDING")
        print("="*60)
        
        print("🔍 What's happening in this phase:")
        print("   1. Loading AI embedding model")
        print("   2. Processing cleaned documents")
        print("   3. Creating vector representations")
        print("   4. Building searchable index")
        print()
        
        try:
            rag = RAGSystem(scraper.output_dir)
            rag_success = await rag.build_rag_index()
            
            if rag_success:
                print("✅ RAG system built successfully!")
                print("\n📊 What was created:")
                print("   • Vector embeddings for semantic search")
                print("   • FAISS index for fast similarity matching")
                print("   • Document chunks ready for AI queries")
                print("   • Knowledge base ready for Q&A")
                
                # Test query
                test_query = input("\nEnter a test question (or press Enter to skip): ").strip()
                if test_query:
                    print("\n🔍 Testing RAG system...")
                    result = await rag.query(test_query)
                    print(f"\n💡 Answer: {result['answer'][:200]}...")
                    print(f"📊 Confidence: {result['confidence']:.2f}")
                    print(f"📚 Sources: {result['documents_found']}")
            else:
                print("❌ Failed to build RAG system")
                print("💡 Common fixes:")
                print("   • Check if cleaned data exists in scraped_data/cleaned/")
                print("   • Ensure OpenAI API key is set in .env")
                print("   • Try removing scraped_data/rag/ and rebuilding")
                
        except Exception as e:
            print(f"❌ RAG system error: {e}")
            print("🔧 Troubleshooting:")
            print("   • Remove old index: rm -rf scraped_data/rag/")
            print("   • Check file permissions")
            print("   • Ensure sufficient disk space")
    
    # PHASE 4: Launch Web Interface
    if launch_web and build_rag:
        print("\n" + "="*60)
        print("PHASE 4: WEB INTERFACE")
        print("="*60)
        
        web_port = os.getenv('WEB_PORT', '8501')
        web_host = os.getenv('WEB_HOST', 'localhost')
        
        print("🌐 Launching web interface...")
        print(f"📱 Open your browser and go to: http://{web_host}:{web_port}")
        print("🔄 Use Ctrl+C to stop the web server")
        
        import subprocess
        import sys
        
        try:
            # Launch Streamlit app with environment variables
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "rag_web_interface.py", 
                "--server.port", web_port,
                "--server.address", web_host
            ])
        except KeyboardInterrupt:
            print("\n🛑 Web interface stopped")
    
    print(f"\n🎉 Complete AI pipeline finished!")
    print(f"📂 Check results in: {scraper.output_dir}")
    print(f"🤖 RAG system ready for queries!")

if __name__ == "__main__":
    asyncio.run(main())