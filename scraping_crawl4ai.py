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
import aiohttp
from dotenv import load_dotenv

# Add PDF support
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.utils import ImageReader
    import fitz  # PyMuPDF for PDF to image conversion
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  PDF libraries not available. Install with: pip install reportlab PyMuPDF")
    PDF_AVAILABLE = False

# Load environment variables
load_dotenv()

class ScreenshotScraper:
    def __init__(self, output_dir="scraped_data", use_proxy=True, output_format="both"):
        self.output_dir = output_dir
        self.visited_urls = set()
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        self.texts_dir = os.path.join(output_dir, "texts")
        self.enhanced_dir = os.path.join(output_dir, "enhanced")
        self.pdfs_dir = os.path.join(output_dir, "pdfs")
        
        # Output format: "screenshot", "pdf", or "both" (default changed to both)
        self.output_format = output_format
        
        # Multiple proxy configurations from environment
        self.use_proxy = use_proxy
        self.proxy_configs = {
            "partners": {
                "server": os.getenv('PROXY_PARTNERS_SERVER', 'http://proxypartners.intratest.com:8080'),
                "username": os.getenv('PROXY_PARTNERS_USERNAME', ''),
                "password": os.getenv('PROXY_PARTNERS_PASSWORD', ''),
                "env_http": f"http://{os.getenv('PROXY_PARTNERS_USERNAME', '')}:{os.getenv('PROXY_PARTNERS_PASSWORD', '')}@{os.getenv('PROXY_PARTNERS_SERVER', 'proxypartners.intratest.com:8080').replace('http://', '')}",
                "env_https": f"http://{os.getenv('PROXY_PARTNERS_USERNAME', '')}:{os.getenv('PROXY_PARTNERS_PASSWORD', '')}@{os.getenv('PROXY_PARTNERS_SERVER', 'proxypartners.intratest.com:8080').replace('http://', '')}"
            },
            "users": {
                "server": os.getenv('PROXY_USERS_SERVER', 'http://proxyusers.intratest.com:8080'),
                "username": os.getenv('PROXY_USERS_USERNAME', ''),
                "password": os.getenv('PROXY_USERS_PASSWORD', ''),
                "env_http": f"http://{os.getenv('PROXY_USERS_USERNAME', '')}:{os.getenv('PROXY_USERS_PASSWORD', '')}@{os.getenv('PROXY_USERS_SERVER', 'proxyusers.intratest.com:8080').replace('http://', '')}",
                "env_https": f"http://{os.getenv('PROXY_USERS_USERNAME', '')}:{os.getenv('PROXY_USERS_PASSWORD', '')}@{os.getenv('PROXY_USERS_SERVER', 'proxyusers.intratest.com:8080').replace('http://', '')}"
            }
        }
        
        self.current_proxy = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        # Create directories
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.texts_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        if PDF_AVAILABLE and output_format in ["pdf", "both"]:
            os.makedirs(self.pdfs_dir, exist_ok=True)
        
        # Configure proxy automatically
        if use_proxy:
            asyncio.create_task(self.setup_best_proxy())
    
    async def test_proxy(self, proxy_name, proxy_config):
        """Test if a proxy is working"""
        try:
            print(f"🔍 Testing proxy {proxy_name}...")
            
            # Configure environment variables temporarily
            original_http = os.environ.get('HTTP_PROXY')
            original_https = os.environ.get('HTTPS_PROXY')
            
            os.environ['HTTP_PROXY'] = proxy_config['env_http']
            os.environ['HTTPS_PROXY'] = proxy_config['env_https']
            
            # Simple connectivity test
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get('http://httpbin.org/ip', timeout=10) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"✅ Proxy {proxy_name} working - IP: {result.get('origin', 'N/A')}")
                            return True
                except:
                    pass
            
            print(f"❌ Proxy {proxy_name} not responding")
            return False
            
        except Exception as e:
            print(f"❌ Error testing proxy {proxy_name}: {e}")
            return False
        finally:
            # Restore environment variables
            if original_http:
                os.environ['HTTP_PROXY'] = original_http
            elif 'HTTP_PROXY' in os.environ:
                del os.environ['HTTP_PROXY']
                
            if original_https:
                os.environ['HTTPS_PROXY'] = original_https
            elif 'HTTPS_PROXY' in os.environ:
                del os.environ['HTTPS_PROXY']
    
    async def setup_best_proxy(self):
        """Automatically choose the best available proxy"""
        print(f"🌐 Detecting best available proxy...")
        
        # Test proxies in order of preference
        proxy_order = ["partners", "users"]  # Priority: partners first
        
        for proxy_name in proxy_order:
            proxy_config = self.proxy_configs[proxy_name]
            
            if await self.test_proxy(proxy_name, proxy_config):
                self.current_proxy = proxy_name
                
                # Configure environment variables
                os.environ['HTTP_PROXY'] = proxy_config['env_http']
                os.environ['HTTPS_PROXY'] = proxy_config['env_https']
                
                print(f"🎯 Proxy selected: {proxy_name} ({proxy_config['server']})")
                return
        
        # If no proxy works, disable
        print(f"⚠️  No proxy available, continuing without proxy")
        self.use_proxy = False
        self.current_proxy = None
    
    def get_current_proxy_config(self):
        """Returns current proxy configuration"""
        if not self.use_proxy or not self.current_proxy:
            return None
        return self.proxy_configs[self.current_proxy]
    
    async def extract_links(self, html_content, base_url):
        """Extract all links from HTML"""
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
            print(f"❌ Error extracting links: {e}")
        
        return links
    
    async def handle_authentication(self, page, url):
        """Handle Murex/Office365 authentication automatically"""
        try:
            print(f"🔐 Checking for authentication requirements...")
            
            # Wait a bit for page to load
            await asyncio.sleep(3)
            
            # Check for various login indicators
            login_indicators = [
                'input[type="email"]',
                'input[name="loginfmt"]',
                'input[placeholder*="email"]',
                'input[placeholder*="Email"]',
                '.sign-in',
                '#signInName',
                '[data-testid="i0116"]'
            ]
            
            login_found = False
            for selector in login_indicators:
                try:
                    if await page.locator(selector).count() > 0:
                        print(f"🔍 Login form detected: {selector}")
                        login_found = True
                        break
                except:
                    continue
            
            if not login_found:
                print(f"✅ No authentication required")
                return True
            
            print(f"🔐 Authentication required - attempting automatic login...")
            
            # Credentials from environment variables
            email = os.getenv('LOGIN_EMAIL', '')
            password = os.getenv('LOGIN_PASSWORD', '')
            
            if not email or not password:
                print(f"❌ Login credentials not found in environment variables")
                return False
            
            # Try different email input selectors
            email_selectors = [
                'input[type="email"]',
                'input[name="loginfmt"]',
                'input[placeholder*="email"]',
                'input[placeholder*="Email"]',
                '#signInName',
                '[data-testid="i0116"]',
                'input[name="username"]',
                'input[id="username"]'
            ]
            
            email_filled = False
            for selector in email_selectors:
                try:
                    email_input = page.locator(selector)
                    if await email_input.count() > 0:
                        print(f"📧 Filling email with selector: {selector}")
                        await email_input.clear()
                        await email_input.fill(email)
                        email_filled = True
                        break
                except Exception as e:
                    print(f"   ❌ Failed with {selector}: {e}")
                    continue
            
            if not email_filled:
                print(f"❌ Could not find email input field")
                return False
            
            # Look for Next/Continue button after email
            next_selectors = [
                'input[type="submit"]',
                'input[value="Next"]',
                'button[type="submit"]',
                '.next-button',
                '#idSIButton9',
                '[data-testid="idSIButton9"]',
                'button:has-text("Next")',
                'button:has-text("Próximo")',
                'input[value="Próximo"]'
            ]
            
            next_clicked = False
            for selector in next_selectors:
                try:
                    next_button = page.locator(selector)
                    if await next_button.count() > 0:
                        print(f"👆 Clicking Next button: {selector}")
                        await next_button.click()
                        next_clicked = True
                        break
                except Exception as e:
                    print(f"   ❌ Failed clicking {selector}: {e}")
                    continue
            
            if next_clicked:
                print(f"⏳ Waiting for password page...")
                await asyncio.sleep(4)
            
            # Now look for password field
            password_selectors = [
                'input[type="password"]',
                'input[name="passwd"]',
                'input[name="password"]',
                '#passwordInput',
                '[data-testid="i0118"]',
                'input[placeholder*="password"]',
                'input[placeholder*="Password"]'
            ]
            
            password_filled = False
            for selector in password_selectors:
                try:
                    password_input = page.locator(selector)
                    if await password_input.count() > 0:
                        print(f"🔑 Filling password with selector: {selector}")
                        await password_input.clear()
                        await password_input.fill(password)
                        password_filled = True
                        break
                except Exception as e:
                    print(f"   ❌ Failed with {selector}: {e}")
                    continue
            
            if not password_filled:
                print(f"❌ Could not find password input field")
                return False
            
            # Submit login form
            submit_selectors = [
                'input[type="submit"]',
                'input[value="Sign in"]',
                'button[type="submit"]',
                '.submit-button',
                '#idSIButton9',
                '[data-testid="idSIButton9"]',
                'button:has-text("Sign in")',
                'button:has-text("Entrar")',
                'input[value="Entrar"]'
            ]
            
            submit_clicked = False
            for selector in submit_selectors:
                try:
                    submit_button = page.locator(selector)
                    if await submit_button.count() > 0:
                        print(f"🚀 Submitting login: {selector}")
                        await submit_button.click()
                        submit_clicked = True
                        break
                except Exception as e:
                    print(f"   ❌ Failed submitting with {selector}: {e}")
                    continue
            
            if not submit_clicked:
                print(f"❌ Could not find submit button")
                return False
            
            # Wait for login to complete
            print(f"⏳ Waiting for authentication to complete...")
            await asyncio.sleep(8)
            
            # Check if we're still on login page or if there are any "Stay signed in" prompts
            stay_signed_selectors = [
                'button:has-text("Yes")',
                'button:has-text("Sim")',
                'input[value="Yes"]',
                'input[value="Sim"]',
                '#idSIButton9',
                'button[data-report-event="Signin_Submit"]',
                'input[type="submit"][value="Yes"]',
                'input[type="submit"][value="Sim"]',
                '.win-button[data-bind*="click: onYes"]',
                'button[class*="primary"]',
                '.ext-button.primary'
            ]
            
            for selector in stay_signed_selectors:
                try:
                    stay_button = page.locator(selector)
                    if await stay_button.count() > 0:
                        print(f"👍 Clicking 'Stay signed in': {selector}")
                        await stay_button.click()
                        await asyncio.sleep(3)
                        break
                except:
                    continue
            
            # Final check - verify we're authenticated
            current_url = page.url
            if "login" not in current_url.lower() and "signin" not in current_url.lower():
                print(f"✅ Authentication successful!")
                print(f"🔗 Current URL: {current_url}")
                return True
            else:
                print(f"⚠️  Still on login page, authentication may have failed")
                print(f"🔗 Current URL: {current_url}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    async def create_pdf_from_page(self, page, filename):
        """Create PDF directly from webpage using Playwright"""
        try:
            if not PDF_AVAILABLE:
                print("❌ PDF libraries not available")
                return None
                
            print(f"📄 Creating PDF from page...")
            
            pdf_path = os.path.join(self.pdfs_dir, f"{filename}.pdf")
            
            # Generate PDF with Playwright (better than screenshot conversion)
            await page.pdf(
                path=pdf_path,
                format='A4',
                print_background=True,
                margin={'top': '1cm', 'right': '1cm', 'bottom': '1cm', 'left': '1cm'},
                prefer_css_page_size=True,
                display_header_footer=False
            )
            
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                file_size = os.path.getsize(pdf_path)
                print(f"✅ PDF created: {file_size} bytes")
                return pdf_path
            else:
                print(f"❌ Failed to create PDF")
                return None
                
        except Exception as e:
            print(f"❌ PDF creation error: {e}")
            return None

    def convert_pdf_to_images(self, pdf_path, filename):
        """Convert PDF pages to high-quality images for OCR"""
        try:
            if not PDF_AVAILABLE:
                return []
                
            print(f"🔄 Converting PDF to images for OCR...")
            
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # High DPI for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = 144 DPI
                pix = page.get_pixmap(matrix=mat)
                
                img_path = os.path.join(self.enhanced_dir, f"{filename}_page_{page_num + 1}.png")
                pix.save(img_path)
                image_paths.append(img_path)
                
                print(f"   📄 Page {page_num + 1} -> {img_path}")
            
            doc.close()
            print(f"✅ PDF converted to {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            print(f"❌ PDF conversion error: {e}")
            return []

    def extract_text_from_pdf_images(self, image_paths):
        """Extract text from PDF-generated images"""
        try:
            print(f"🔍 Extracting text from PDF images...")
            
            all_text = ""
            total_confidence = 0
            page_count = 0
            
            for i, img_path in enumerate(image_paths):
                print(f"   📄 Processing page {i + 1}/{len(image_paths)}...")
                
                # OCR configuration optimized for PDF-generated images
                config = r'--oem 3 --psm 1 -l por+eng -c preserve_interword_spaces=1'
                
                try:
                    # Calculate confidence
                    data = pytesseract.image_to_data(
                        Image.open(img_path), 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Extract text
                    page_text = pytesseract.image_to_string(Image.open(img_path), config=config)
                    
                    if page_text.strip():
                        all_text += f"\n--- PAGE {i + 1} ---\n{page_text}\n"
                        total_confidence += avg_confidence
                        page_count += 1
                        
                        print(f"      ✅ {len(page_text)} chars, confidence: {avg_confidence:.1f}%")
                    else:
                        print(f"      ⚠️  No text extracted")
                        
                except Exception as page_error:
                    print(f"      ❌ Error: {page_error}")
                    continue
            
            final_confidence = total_confidence / page_count if page_count > 0 else 0
            
            if all_text.strip():
                # Clean text
                final_text = re.sub(r'\s+', ' ', all_text.strip())
                final_text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ\n]', '', final_text)
                
                print(f"🏆 PDF OCR completed!")
                print(f"   📄 Pages processed: {page_count}")
                print(f"   📝 Total characters: {len(final_text)}")
                print(f"   🎯 Average confidence: {final_confidence:.1f}%")
                
                return final_text
            else:
                print("❌ No text extracted from PDF")
                return ""
                
        except Exception as e:
            print(f"❌ PDF OCR error: {e}")
            return ""

    async def take_screenshot_playwright(self, url, filename):
        """Take screenshot and/or create PDF based on output_format"""
        try:
            print(f"📸 Capturing content from: {url}")
            print(f"📋 Output format: {self.output_format}")
            
            if self.use_proxy and self.current_proxy:
                print(f"🌐 Using proxy: {self.current_proxy} ({self.proxy_configs[self.current_proxy]['server']})")
            
            async with async_playwright() as p:
                # Configure browser with simpler settings to avoid proxy issues
                browser_args = [
                    '--no-sandbox', 
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
                
                proxy_config = self.get_current_proxy_config()
                
                # Try without persistent context first for better proxy compatibility
                try:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=browser_args,
                        timeout=60000
                    )
                    
                    context = await browser.new_context(
                        viewport={'width': 1280, 'height': 720},
                        device_scale_factor=1,
                        ignore_https_errors=True,
                        proxy={
                            "server": proxy_config["server"],
                            "username": proxy_config["username"],
                            "password": proxy_config["password"]
                        } if proxy_config else None
                    )
                    
                    page = await context.new_page()
                    
                except Exception as browser_error:
                    print(f"⚠️  Proxy browser launch failed: {browser_error}")
                    print("🔄 Trying without proxy...")
                    
                    # Fallback: try without proxy
                    browser = await p.chromium.launch(
                        headless=True,
                        args=browser_args,
                        timeout=60000
                    )
                    
                    context = await browser.new_context(
                        viewport={'width': 1280, 'height': 720},
                        device_scale_factor=1,
                        ignore_https_errors=True
                    )
                    
                    page = await context.new_page()
                    self.use_proxy = False  # Disable proxy for this session
                
                try:
                    # Set shorter timeouts
                    page.set_default_timeout(30000)
                    
                    # Configure headers
                    await page.set_extra_http_headers({
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive'
                    })
                    
                    print(f"🔗 Navigating to: {url}")
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    print(f"✅ Page loaded successfully")
                    
                    # Skip authentication for Wikipedia
                    if "wikipedia.org" not in url.lower():
                        login_check = await page.locator('input[type="email"], input[name="loginfmt"], #signInName').count()
                        if login_check > 0:
                            auth_success = await self.handle_authentication(page, url)
                            if not auth_success:
                                print(f"⚠️  Authentication failed, but continuing...")
                    else:
                        print(f"✅ No authentication needed for Wikipedia")
                    
                    # Simple content loading
                    await asyncio.sleep(2)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    await asyncio.sleep(1)
                    await page.evaluate("window.scrollTo(0, 0);")
                    await asyncio.sleep(1)
                    
                    screenshot_path = None
                    pdf_path = None
                    html_content = await page.content()  # Keep for link extraction only
                    
                    # Create screenshot if requested
                    if self.output_format in ["screenshot", "both"]:
                        try:
                            screenshot_path = os.path.join(self.screenshots_dir, f"{filename}.png")
                            await page.screenshot(
                                path=screenshot_path, 
                                full_page=True,
                                timeout=30000  # Increased timeout to 30 seconds
                            )
                            
                            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                                file_size = os.path.getsize(screenshot_path)
                                print(f"✅ Screenshot saved: {file_size} bytes")
                            else:
                                print(f"❌ Screenshot file empty or not created")
                                screenshot_path = None
                        except Exception as screenshot_error:
                            print(f"⚠️  Screenshot failed: {screenshot_error}")
                            screenshot_path = None
                    
                    # Create PDF if requested
                    if self.output_format in ["pdf", "both"] and PDF_AVAILABLE:
                        try:
                            pdf_path = await self.create_pdf_from_page(page, filename)
                        except Exception as pdf_error:
                            print(f"⚠️  PDF creation failed: {pdf_error}")
                            pdf_path = None
                    
                    await browser.close()
                    
                    return screenshot_path, pdf_path, html_content
                    
                except Exception as page_error:
                    print(f"❌ Error processing page: {page_error}")
                    await browser.close()
                    return None, None, None
                    
        except Exception as e:
            print(f"❌ Playwright error: {e}")
            return None, None, None
    
    async def scrape_url(self, url, max_depth=2, current_depth=0):
        """Scrape a URL and its sub-pages"""
        if current_depth >= max_depth or url in self.visited_urls:
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 Processing (depth {current_depth}): {url}")
        print(f"{'='*80}")
        
        self.visited_urls.add(url)
        
        # Generate unique filename
        parsed_url = urlparse(url)
        filename = f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}_{len(self.visited_urls)}"
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))[:100]
        
        try:
            start_time = time.time()
            
            # 1. Take screenshot and/or PDF
            screenshot_path, pdf_path, html_content = await self.take_screenshot_playwright(url, filename)
            
            # 2. Check if we got any content via screenshot or PDF
            if (screenshot_path and os.path.exists(screenshot_path)) or (pdf_path and os.path.exists(pdf_path)):
                print(f"\n🔍 Processing OCR content (format: {self.output_format})...")
                
                # Process in separate thread
                loop = asyncio.get_event_loop()
                ocr_text = await loop.run_in_executor(
                    self.executor, 
                    self.process_content_and_ocr, 
                    screenshot_path, 
                    pdf_path,
                    filename
                )
                
                if ocr_text and ocr_text.strip():
                    # 3. Save text with metadata
                    processing_time = time.time() - start_time
                    proxy_info = f"{self.current_proxy}" if self.current_proxy else "Disabled"
                    
                    full_text = f"URL: {url}\n"
                    full_text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    full_text += f"Depth: {current_depth}\n"
                    full_text += f"Method: OCR + PDF ({self.output_format})\n"
                    full_text += f"Proxy: {proxy_info}\n"
                    full_text += f"Time: {processing_time:.1f}s\n"
                    full_text += f"Characters: {len(ocr_text)}\n"
                    full_text += f"Words: {len(ocr_text.split())}\n"
                    full_text += "=" * 50 + "\n\n"
                    full_text += ocr_text
                    
                    # Save text
                    text_path = await self.save_text(full_text, filename)
                    print(f"💾 Text saved in {processing_time:.1f}s: {text_path}")
                else:
                    print("⚠️  Could not extract text from OCR")
            else:
                print("❌ Could not capture any content (screenshot or PDF failed)")
            
            # Process next level links using HTML (for navigation only)
            if current_depth < max_depth - 1 and html_content:
                links = await self.extract_links(html_content, url)
                valid_links = [link for link in links if link not in self.visited_urls]
                print(f"🔗 Found {len(valid_links)} new links")
                
                # Limit links to process (avoid too many)
                max_links = min(3, len(valid_links))  # Process max 3 links per page
                for link in valid_links[:max_links]:
                    await self.scrape_url(link, max_depth, current_depth + 1)
                    await asyncio.sleep(2)  # Rate limiting
        
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")

    async def run(self, start_url, max_depth=2):
        """Execute scraping"""
        try:
            proxy_status = "🌐 AUTO PROXY" if self.use_proxy else "🚫 NO PROXY"
            
            # Enhanced mode description
            mode_descriptions = {
                "screenshot": "📸 Screenshot + Enhanced Image + OCR",
                "pdf": "📄 PDF + High-Quality OCR",
                "both": "🚀 Screenshot + PDF + Smart OCR Selection"
            }
            
            print(f"🚀 Scraping with Enhanced OCR {proxy_status}")
            print(f"🎯 Initial URL: {start_url}")
            print(f"📊 Maximum depth: {max_depth}")
            print(f"💾 Data will be saved to: {self.output_dir}")
            
            if self.use_proxy:
                # Wait for proxy configuration
                await self.setup_best_proxy()
                if self.current_proxy:
                    print(f"🌐 Active proxy: {self.current_proxy}")
            
            print(f"📸 Mode: {mode_descriptions[self.output_format]}")
            
            await self.scrape_url(start_url, max_depth)
            
            # Close executor
            self.executor.shutdown(wait=True)
            
            print(f"\n✅ Scraping completed!")
            print(f"📄 Total pages processed: {len(self.visited_urls)}")
            
            if self.output_format in ["screenshot", "both"]:
                print(f"📸 Original screenshots: {self.screenshots_dir}")
                print(f"🔧 Enhanced images: {self.enhanced_dir}")
            
            if self.output_format in ["pdf", "both"]:
                print(f"📄 PDF files: {self.pdfs_dir}")
            
            print(f"📝 Extracted texts: {self.texts_dir}")
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            self.executor.shutdown(wait=False)

    async def save_text(self, text, filename):
        """Save extracted text to file"""
        try:
            text_path = os.path.join(self.texts_dir, f"{filename}.txt")
            async with aiofiles.open(text_path, 'w', encoding='utf-8') as f:
                await f.write(text)
            return text_path
        except Exception as e:
            print(f"❌ Error saving text: {e}")
            return None
    
    def create_enhanced_image(self, image_path, filename):
        """Create ONLY the enhanced version (sharper)"""
        try:
            print(f"🔧 Creating enhanced image...")
            
            # Load original image
            original_img = Image.open(image_path)
            
            # Apply quality improvements
            enhanced = ImageEnhance.Contrast(original_img).enhance(1.3)  # More contrast
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)     # More sharpness
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)    # Slightly more brightness
            
            # Save enhanced image
            enhanced_path = os.path.join(self.enhanced_dir, f"{filename}_enhanced.png")
            enhanced.save(enhanced_path, 'PNG', optimize=True, dpi=(300, 300))
            
            print(f"✅ Enhanced image created: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            print(f"❌ Error creating enhanced image: {e}")
            return image_path  # Return original if it fails
    
    def extract_text_from_enhanced(self, enhanced_image_path):
        """Extract text ONLY from enhanced image"""
        try:
            print(f"🔍 Extracting text from enhanced image...")
            
            if not os.path.exists(enhanced_image_path):
                print(f"❌ Enhanced image not found: {enhanced_image_path}")
                return ""
            
            # OCR configurations optimized for enhanced image
            configs = [
                r'--oem 3 --psm 3 -l por+eng -c preserve_interword_spaces=1',  # Full page
                r'--oem 3 --psm 6 -l por+eng -c preserve_interword_spaces=1',  # Text block
                r'--oem 3 --psm 1 -l por+eng',  # Auto orientation
            ]
            
            best_text = ""
            best_confidence = 0
            
            for i, config in enumerate(configs):
                try:
                    print(f"🔍 Testing config {i+1}/3...")
                    
                    # Calculate confidence
                    data = pytesseract.image_to_data(
                        Image.open(enhanced_image_path), 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Extract text
                    text = pytesseract.image_to_string(Image.open(enhanced_image_path), config=config)
                    
                    char_count = len(text.strip())
                    print(f"   📊 Config {i+1}: {char_count} chars, confidence: {avg_confidence:.1f}%")
                    
                    # Use best result
                    if avg_confidence > best_confidence and char_count > 20:
                        best_text = text
                        best_confidence = avg_confidence
                        print(f"   ✅ New best result!")
                        
                    # If result is very good, stop
                    if avg_confidence > 85 and char_count > 300:
                        print(f"   🎯 Excellent result, stopping...")
                        break
                        
                except Exception as config_error:
                    print(f"   ❌ Config {i+1} error: {config_error}")
                    continue
            
            if best_text.strip():
                # Clean text
                final_text = re.sub(r'\s+', ' ', best_text.strip())
                final_text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ]', '', final_text)
                
                print(f"🏆 Text extracted successfully!")
                print(f"   📝 Characters: {len(final_text)}")
                print(f"   🎯 Confidence: {best_confidence:.1f}%")
                print(f"   📖 Preview: {final_text[:150]}...")
                
                return final_text
            else:
                print("❌ No text extracted")
                return ""
            
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return ""
    
    def extract_text_from_pdf_direct(self, pdf_path):
        """Extract text directly from PDF without OCR (faster for text-based PDFs)"""
        try:
            if not PDF_AVAILABLE:
                print("❌ PDF libraries not available")
                return ""
                
            print(f"📄 Extracting text directly from PDF...")
            
            doc = fitz.open(pdf_path)
            all_text = ""
            total_chars = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    all_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                    page_chars = len(page_text)
                    total_chars += page_chars
                    print(f"   📄 Page {page_num + 1}: {page_chars} characters")
                else:
                    print(f"   📄 Page {page_num + 1}: No text found (might be image-based)")
            
            doc.close()
            
            if all_text.strip():
                # Clean text
                final_text = re.sub(r'\s+', ' ', all_text.strip())
                final_text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ]', '', final_text)
                
                print(f"🏆 PDF text extraction completed!")
                print(f"   📄 Pages processed: {len(doc)}")
                print(f"   📝 Total characters: {len(final_text)}")
                print(f"   📖 Preview: {final_text[:150]}...")
                
                return final_text
            else:
                print("❌ No text extracted from PDF (might be image-based)")
                return ""
                
        except Exception as e:
            print(f"❌ PDF text extraction error: {e}")
            return ""

    def process_content_and_ocr(self, screenshot_path, pdf_path, filename):
        """Process both screenshot and PDF for OCR comparison"""
        try:
            screenshot_text = ""
            pdf_text = ""
            pdf_direct_text = ""
            
            # Process screenshot if available
            if screenshot_path and os.path.exists(screenshot_path):
                print(f"\n🔍 Processing Screenshot OCR...")
                enhanced_path = self.create_enhanced_image(screenshot_path, filename)
                screenshot_text = self.extract_text_from_enhanced(enhanced_path)
            
            # Process PDF if available
            if pdf_path and os.path.exists(pdf_path):
                # Try direct text extraction first (faster)
                print(f"\n📄 Processing PDF Direct Text Extraction...")
                pdf_direct_text = self.extract_text_from_pdf_direct(pdf_path)
                
                # If direct extraction fails or gives poor results, use OCR
                if not pdf_direct_text or len(pdf_direct_text.strip()) < 100:
                    print(f"\n📄 Direct extraction failed/poor, trying PDF OCR...")
                    pdf_images = self.convert_pdf_to_images(pdf_path, filename)
                    if pdf_images:
                        pdf_text = self.extract_text_from_pdf_images(pdf_images)
                else:
                    print(f"✅ Direct PDF extraction successful, skipping OCR")
                    pdf_text = pdf_direct_text
            
            # Choose best result or combine
            if self.output_format == "both":
                results = []
                scores = []
                
                if screenshot_text:
                    screenshot_score = len(screenshot_text.strip())
                    results.append(("Screenshot OCR", screenshot_text, screenshot_score))
                    scores.append(screenshot_score)
                
                if pdf_text:
                    pdf_score = len(pdf_text.strip())
                    results.append(("PDF Text", pdf_text, pdf_score))
                    scores.append(pdf_score)
                
                if results:
                    print(f"\n📊 Text Extraction Comparison:")
                    for method, text, score in results:
                        print(f"   📄 {method}: {score} characters")
                    
                    # Use the method with most content
                    best_result = max(results, key=lambda x: x[2])
                    print(f"   🏆 Using {best_result[0]} (best quality)")
                    return best_result[1]
            
            elif self.output_format == "pdf":
                return pdf_text
            elif self.output_format == "screenshot":
                return screenshot_text
            
            # Fallback: return any available text
            return pdf_text or screenshot_text or ""
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return ""

async def main():
    """Main function"""
    # Load default URL from environment
    default_url = os.getenv('DEFAULT_URL', 'https://example.com')
    url = input(f"Enter initial URL (default: {default_url}): ").strip()
    if not url:
        url = default_url
        print(f"Using default URL: {url}")
    
    # Load default depth from environment
    default_depth = os.getenv('DEFAULT_MAX_DEPTH', '2')
    depth = input(f"Enter maximum depth (default: {default_depth}): ").strip()
    try:
        max_depth = int(depth) if depth else int(default_depth)
    except ValueError:
        max_depth = int(default_depth)
    
    # Ask about proxy with environment default
    use_proxy_default = os.getenv('USE_PROXY', 'true').lower() in ['true', '1', 'yes', 'on']
    use_proxy_input = input(f"Use corporate proxy? ({'Y' if use_proxy_default else 'n'}/{'n' if use_proxy_default else 'Y'}): ").strip().lower()
    use_proxy = use_proxy_input not in ['n', 'no'] if use_proxy_input else use_proxy_default
    
    # Ask about output format with new default
    print("\n📋 Choose capture format:")
    print("1. Screenshot only (traditional)")
    print("2. PDF only (better for text-heavy pages)")
    print("3. 🚀 Both (Smart OCR - best quality) [RECOMMENDED]")
    
    format_choice = input("Choose format (1/2/3, default: 3): ").strip()
    
    if format_choice == "1":
        output_format = "screenshot"
    elif format_choice == "2":
        output_format = "pdf"
    else:  # Default to "both" (option 3)
        output_format = "both"
    
    print(f"✅ Selected format: {output_format}")
    
    scraper = ScreenshotScraper(use_proxy=use_proxy, output_format=output_format)
    await scraper.run(url, max_depth=max_depth)

if __name__ == "__main__":
    asyncio.run(main())