import os
import time
import asyncio
from urllib.parse import urljoin, urlparse
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import aiofiles
from bs4 import BeautifulSoup
import re
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class ScreenshotScraper:
    def __init__(self, output_dir="scraped_data", use_proxy=True):
        self.output_dir = output_dir
        self.visited_urls = set()
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        self.texts_dir = os.path.join(output_dir, "texts")
        self.enhanced_dir = os.path.join(output_dir, "enhanced")
        
        # Configurações de proxy
        self.use_proxy = use_proxy
        self.proxy_config = {
            "server": "http://proxypartners.intranatixis.com:8080",
            "username": "cardosoti",
            "password": "Sucesso2025+Total"
        } if use_proxy else None
        
        # Pool de threads para processamento paralelo
        self.executor = ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        # Criar diretórios
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.texts_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        
        # Configurar variáveis de ambiente para proxy
        if use_proxy:
            os.environ['HTTP_PROXY'] = f"http://cardosoti:Sucesso2025+Total@proxypartners.intranatixis.com:8080"
            os.environ['HTTPS_PROXY'] = f"http://cardosoti:Sucesso2025+Total@proxypartners.intranatixis.com:8080"
            print(f"🌐 Proxy configurado: proxypartners.intranatixis.com:8080")
    
    async def extract_links(self, html_content, base_url):
        """Extrai todos os links do HTML"""
        links = set()
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            link_elements = soup.find_all('a', href=True)
            
            for link in link_elements:
                href = link['href']
                if href:
                    absolute_url = urljoin(base_url, href)
                    if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                        clean_url = absolute_url.split('#')[0]
                        links.add(clean_url)
            
        except Exception as e:
            print(f"❌ Erro ao extrair links: {e}")
        
        return links
    
    async def take_screenshot_playwright(self, url, filename):
        """Tira screenshot de alta qualidade usando Playwright com proxy"""
        try:
            print(f"📸 Capturando screenshot de: {url}")
            if self.use_proxy:
                print(f"🌐 Usando proxy: proxypartners.intranatixis.com:8080")
            
            async with async_playwright() as p:
                # Configurar browser com proxy
                browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
                
                if self.use_proxy:
                    browser_args.extend([
                        f'--proxy-server={self.proxy_config["server"]}',
                        '--disable-web-security',
                        '--ignore-certificate-errors',
                        '--ignore-ssl-errors'
                    ])
                
                browser = await p.chromium.launch(
                    headless=True,
                    args=browser_args
                )
                
                # Configurar contexto com proxy
                context_config = {
                    'viewport': {'width': 1920, 'height': 1080},
                    'device_scale_factor': 2,
                    'ignore_https_errors': True
                }
                
                if self.use_proxy:
                    context_config['proxy'] = self.proxy_config
                
                context = await browser.new_context(**context_config)
                
                page = await context.new_page()
                
                try:
                    # Configurar headers para bypass de detecção
                    await page.set_extra_http_headers({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    })
                    
                    print(f"🔗 Navegando para: {url}")
                    await page.goto(url, wait_until='networkidle', timeout=60000)
                    print(f"✅ Página carregada com sucesso")
                    
                    await asyncio.sleep(3)
                    
                    # Scroll para carregar conteúdo
                    print(f"📜 Fazendo scroll para carregar conteúdo...")
                    await page.evaluate("""
                        async () => {
                            window.scrollTo(0, document.body.scrollHeight);
                            await new Promise(resolve => setTimeout(resolve, 2000));
                            window.scrollTo(0, 0);
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }
                    """)
                    
                    screenshot_path = os.path.join(self.screenshots_dir, f"{filename}.png")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    
                    html_content = await page.content()
                    await browser.close()
                    
                    if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                        file_size = os.path.getsize(screenshot_path)
                        print(f"✅ Screenshot salvo: {file_size} bytes")
                        return screenshot_path, html_content
                    else:
                        print(f"❌ Falha ao criar screenshot")
                        return None, html_content
                        
                except Exception as page_error:
                    print(f"❌ Erro ao processar página: {page_error}")
                    await browser.close()
                    return None, None
                    
        except Exception as e:
            print(f"❌ Erro no Playwright: {e}")
            return None, None
    
    def create_enhanced_image(self, image_path, filename):
        """Cria APENAS a versão enhanced (mais nítida)"""
        try:
            print(f"🔧 Criando imagem enhanced...")
            
            # Carregar imagem original
            original_img = Image.open(image_path)
            
            # Aplicar melhorias de qualidade
            enhanced = ImageEnhance.Contrast(original_img).enhance(1.3)  # Mais contraste
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)     # Mais nitidez
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)    # Pouco mais brilho
            
            # Salvar imagem enhanced
            enhanced_path = os.path.join(self.enhanced_dir, f"{filename}_enhanced.png")
            enhanced.save(enhanced_path, 'PNG', optimize=True, dpi=(300, 300))
            
            print(f"✅ Imagem enhanced criada: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            print(f"❌ Erro ao criar imagem enhanced: {e}")
            return image_path  # Retorna original se falhar
    
    def extract_text_from_enhanced(self, enhanced_image_path):
        """Extrai texto APENAS da imagem enhanced"""
        try:
            print(f"🔍 Extraindo texto da imagem enhanced...")
            
            if not os.path.exists(enhanced_image_path):
                print(f"❌ Imagem enhanced não encontrada: {enhanced_image_path}")
                return ""
            
            # Configurações OCR otimizadas para imagem enhanced
            configs = [
                r'--oem 3 --psm 3 -l por+eng -c preserve_interword_spaces=1',  # Página completa
                r'--oem 3 --psm 6 -l por+eng -c preserve_interword_spaces=1',  # Bloco de texto
                r'--oem 3 --psm 1 -l por+eng',  # Auto orientação
            ]
            
            best_text = ""
            best_confidence = 0
            
            for i, config in enumerate(configs):
                try:
                    print(f"🔍 Testando config {i+1}/3...")
                    
                    # Calcular confiança
                    data = pytesseract.image_to_data(
                        Image.open(enhanced_image_path), 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Extrair texto
                    text = pytesseract.image_to_string(Image.open(enhanced_image_path), config=config)
                    
                    char_count = len(text.strip())
                    print(f"   📊 Config {i+1}: {char_count} chars, confiança: {avg_confidence:.1f}%")
                    
                    # Usar melhor resultado
                    if avg_confidence > best_confidence and char_count > 20:
                        best_text = text
                        best_confidence = avg_confidence
                        print(f"   ✅ Novo melhor resultado!")
                        
                    # Se resultado é muito bom, parar
                    if avg_confidence > 85 and char_count > 300:
                        print(f"   🎯 Resultado excelente, parando...")
                        break
                        
                except Exception as config_error:
                    print(f"   ❌ Erro config {i+1}: {config_error}")
                    continue
            
            if best_text.strip():
                # Limpar texto
                final_text = re.sub(r'\s+', ' ', best_text.strip())
                final_text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ]', '', final_text)
                
                print(f"🏆 Texto extraído com sucesso!")
                print(f"   📝 Caracteres: {len(final_text)}")
                print(f"   🎯 Confiança: {best_confidence:.1f}%")
                print(f"   📖 Preview: {final_text[:150]}...")
                
                return final_text
            else:
                print("❌ Nenhum texto extraído")
                return ""
            
        except Exception as e:
            print(f"❌ Erro na extração de texto: {e}")
            return ""
    
    async def save_text(self, text, filename):
        """Salva o texto extraído em arquivo"""
        try:
            text_path = os.path.join(self.texts_dir, f"{filename}.txt")
            async with aiofiles.open(text_path, 'w', encoding='utf-8') as f:
                await f.write(text)
            return text_path
        except Exception as e:
            print(f"❌ Erro ao salvar texto: {e}")
            return None
    
    def process_image_and_ocr(self, screenshot_path, filename):
        """Processa imagem e OCR em thread separada"""
        try:
            # 1. Criar imagem enhanced
            enhanced_path = self.create_enhanced_image(screenshot_path, filename)
            
            # 2. Extrair texto da imagem enhanced
            extracted_text = self.extract_text_from_enhanced(enhanced_path)
            
            return extracted_text
            
        except Exception as e:
            print(f"❌ Erro no processamento: {e}")
            return ""
    
    async def scrape_url(self, url, max_depth=2, current_depth=0):
        """Faz scraping de uma URL e suas sub-páginas"""
        if current_depth >= max_depth or url in self.visited_urls:
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 Processando (depth {current_depth}): {url}")
        print(f"{'='*80}")
        
        self.visited_urls.add(url)
        
        # Gerar nome de arquivo único
        parsed_url = urlparse(url)
        filename = f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}_{len(self.visited_urls)}"
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))[:100]
        
        try:
            start_time = time.time()
            
            # 1. Tirar screenshot
            screenshot_path, html_content = await self.take_screenshot_playwright(url, filename)
            
            if screenshot_path and os.path.exists(screenshot_path):
                print("\n🔍 Processando imagem enhanced + OCR...")
                
                # 2. Processar imagem e OCR em thread separada
                loop = asyncio.get_event_loop()
                ocr_text = await loop.run_in_executor(
                    self.executor, 
                    self.process_image_and_ocr, 
                    screenshot_path, 
                    filename
                )
                
                if ocr_text.strip():
                    # 3. Salvar texto com metadados
                    processing_time = time.time() - start_time
                    full_text = f"URL: {url}\n"
                    full_text += f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    full_text += f"Profundidade: {current_depth}\n"
                    full_text += f"Método: Enhanced OCR\n"
                    full_text += f"Proxy: {'Ativo' if self.use_proxy else 'Desabilitado'}\n"
                    full_text += f"Tempo: {processing_time:.1f}s\n"
                    full_text += f"Caracteres: {len(ocr_text)}\n"
                    full_text += f"Palavras: {len(ocr_text.split())}\n"
                    full_text += "=" * 50 + "\n\n"
                    full_text += ocr_text
                    
                    # Salvar texto
                    text_path = await self.save_text(full_text, filename)
                    print(f"💾 Texto salvo em {processing_time:.1f}s: {text_path}")
                else:
                    print("⚠️  Não foi possível extrair texto")
            else:
                print("❌ Não foi possível capturar screenshot")
            
            # Processar links do próximo nível
            if current_depth < max_depth - 1 and html_content:
                links = await self.extract_links(html_content, url)
                valid_links = [link for link in links if link not in self.visited_urls]
                print(f"🔗 Encontrados {len(valid_links)} novos links")
                
                # Processar alguns links
                for link in valid_links[:2]:
                    await self.scrape_url(link, max_depth, current_depth + 1)
                    await asyncio.sleep(2)
        
        except Exception as e:
            print(f"❌ Erro ao processar {url}: {e}")
    
    async def run(self, start_url, max_depth=2):
        """Executa o scraping"""
        try:
            proxy_status = "🌐 COM PROXY" if self.use_proxy else "🚫 SEM PROXY"
            print(f"🚀 Scraping com Enhanced OCR {proxy_status}")
            print(f"🎯 URL inicial: {start_url}")
            print(f"📊 Profundidade máxima: {max_depth}")
            print(f"💾 Dados serão salvos em: {self.output_dir}")
            if self.use_proxy:
                print(f"🌐 Proxy: proxypartners.intranatixis.com:8080")
            print("📸 Modo: Screenshot + Enhanced Image + OCR")
            
            await self.scrape_url(start_url, max_depth)
            
            # Fechar executor
            self.executor.shutdown(wait=True)
            
            print(f"\n✅ Scraping concluído!")
            print(f"📄 Total de páginas processadas: {len(self.visited_urls)}")
            print(f"📸 Screenshots originais: {self.screenshots_dir}")
            print(f"🔧 Imagens enhanced: {self.enhanced_dir}")
            print(f"📝 Textos extraídos: {self.texts_dir}")
            
        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
            self.executor.shutdown(wait=False)

async def main():
    """Função principal"""
    url = input("Digite a URL inicial: ").strip()
    if not url:
        url = "https://example.com"
        print(f"Usando URL padrão: {url}")
    
    depth = input("Digite a profundidade máxima (padrão 2): ").strip()
    try:
        max_depth = int(depth) if depth else 2
    except ValueError:
        max_depth = 2
    
    # Perguntar sobre proxy
    use_proxy_input = input("Usar proxy corporativo? (s/N): ").strip().lower()
    use_proxy = use_proxy_input in ['s', 'sim', 'y', 'yes']
    
    scraper = ScreenshotScraper(use_proxy=use_proxy)
    await scraper.run(url, max_depth=max_depth)

if __name__ == "__main__":
    asyncio.run(main())