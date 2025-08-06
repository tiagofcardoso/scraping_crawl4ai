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

class ScreenshotScraper:
    def __init__(self, output_dir="scraped_data", use_proxy=True):
        self.output_dir = output_dir
        self.visited_urls = set()
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        self.texts_dir = os.path.join(output_dir, "texts")
        self.enhanced_dir = os.path.join(output_dir, "enhanced")
        
        # Multiple proxy configurations
        self.use_proxy = use_proxy
        self.proxy_configs = {
            "partners": {
                "server": "http://proxypartners.intranatixis.com:8080",
                "username": "cardosoti",
                "password": "Sucesso2025+Total",
                "env_http": "http://cardosoti:Sucesso2025+Total@proxypartners.intranatixis.com:8080",
                "env_https": "http://cardosoti:Sucesso2025+Total@proxypartners.intranatixis.com:8080"
            },
            "users": {
                "server": "http://proxyusers.intranatixis.com:8080",
                "username": "cardosoti",
                "password": "Sucesso2025+Total",
                "env_http": "http://cardosoti:Sucesso2025+Total@proxyusers.intranatixis.com:8080",
                "env_https": "http://cardosoti:Sucesso2025+Total@proxyusers.intranatixis.com:8080"
            }
        }
        
        self.current_proxy = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        # Create directories
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.texts_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        
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
            
            # Credentials
            email = "tiago.cardoso@natixis.com"
            password = "Sucesso2025+Total"
            
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
                '#idSIButton9'
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

    async def take_screenshot_playwright(self, url, filename):
        """Take high-quality screenshot using Playwright with automatic proxy and authentication"""
        try:
            print(f"📸 Capturing screenshot of: {url}")
            
            if self.use_proxy and self.current_proxy:
                print(f"🌐 Using proxy: {self.current_proxy} ({self.proxy_configs[self.current_proxy]['server']})")
            
            async with async_playwright() as p:
                # Configure browser with proxy
                browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
                
                proxy_config = self.get_current_proxy_config()
                if proxy_config:
                    browser_args.extend([
                        f'--proxy-server={proxy_config["server"]}',
                        '--disable-web-security',
                        '--ignore-certificate-errors',
                        '--ignore-ssl-errors'
                    ])
                
                browser = await p.chromium.launch(
                    headless=True,
                    args=browser_args
                )
                
                # Configure context with proxy
                context_config = {
                    'viewport': {'width': 1920, 'height': 1080},
                    'device_scale_factor': 2,
                    'ignore_https_errors': True
                }
                
                if proxy_config:
                    context_config['proxy'] = {
                        "server": proxy_config["server"],
                        "username": proxy_config["username"],
                        "password": proxy_config["password"]
                    }
                
                context = await browser.new_context(**context_config)
                
                page = await context.new_page()
                
                try:
                    # Configure headers for detection bypass
                    await page.set_extra_http_headers({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9,pt;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    })
                    
                    print(f"🔗 Navigating to: {url}")
                    await page.goto(url, wait_until='networkidle', timeout=60000)
                    print(f"✅ Page loaded successfully")
                    
                    # Handle authentication if required
                    auth_success = await self.handle_authentication(page, url)
                    if not auth_success:
                        print(f"⚠️  Authentication failed, but continuing...")
                    
                    await asyncio.sleep(3)
                    
                    # Scroll to load content
                    print(f"📜 Scrolling to load content...")
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
                        print(f"✅ Screenshot saved: {file_size} bytes")
                        return screenshot_path, html_content
                    else:
                        print(f"❌ Failed to create screenshot")
                        return None, html_content
                        
                except Exception as page_error:
                    print(f"❌ Error processing page: {page_error}")
                    
                    # If it fails, try switching proxy
                    if self.use_proxy and self.current_proxy:
                        print(f"🔄 Trying to switch proxy...")
                        await self.try_alternative_proxy()
                    
                    await browser.close()
                    return None, None
                    
        except Exception as e:
            print(f"❌ Playwright error: {e}")
            return None, None
    
    async def try_alternative_proxy(self):
        """Try using alternative proxy if current one fails"""
        current = self.current_proxy
        
        if current == "partners":
            alternative = "users"
        elif current == "users":
            alternative = "partners"
        else:
            return
        
        print(f"🔄 Trying alternative proxy: {alternative}")
        
        if await self.test_proxy(alternative, self.proxy_configs[alternative]):
            self.current_proxy = alternative
            alt_config = self.proxy_configs[alternative]
            
            # Update environment variables
            os.environ['HTTP_PROXY'] = alt_config['env_http']
            os.environ['HTTPS_PROXY'] = alt_config['env_https']
            
            print(f"✅ Switched to alternative proxy: {alternative}")
        else:
            print(f"❌ Alternative proxy {alternative} also not working")
    
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
    
    def process_image_and_ocr(self, screenshot_path, filename):
        """Process image and OCR in separate thread"""
        try:
            # 1. Create enhanced image
            enhanced_path = self.create_enhanced_image(screenshot_path, filename)
            
            # 2. Extract text from enhanced image
            extracted_text = self.extract_text_from_enhanced(enhanced_path)
            
            return extracted_text
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return ""
    
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
            
            # 1. Take screenshot
            screenshot_path, html_content = await self.take_screenshot_playwright(url, filename)
            
            if screenshot_path and os.path.exists(screenshot_path):
                print("\n🔍 Processing enhanced image + OCR...")
                
                # 2. Process image and OCR in separate thread
                loop = asyncio.get_event_loop()
                ocr_text = await loop.run_in_executor(
                    self.executor, 
                    self.process_image_and_ocr, 
                    screenshot_path, 
                    filename
                )
                
                if ocr_text.strip():
                    # 3. Save text with metadata
                    processing_time = time.time() - start_time
                    proxy_info = f"{self.current_proxy}" if self.current_proxy else "Disabled"
                    
                    full_text = f"URL: {url}\n"
                    full_text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    full_text += f"Depth: {current_depth}\n"
                    full_text += f"Method: Enhanced OCR\n"
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
                    print("⚠️  Could not extract text")
            else:
                print("❌ Could not capture screenshot")
            
            # Process next level links
            if current_depth < max_depth - 1 and html_content:
                links = await self.extract_links(html_content, url)
                valid_links = [link for link in links if link not in self.visited_urls]
                print(f"🔗 Found {len(valid_links)} new links")
                
                # Process some links
                for link in valid_links[:2]:
                    await self.scrape_url(link, max_depth, current_depth + 1)
                    await asyncio.sleep(2)
        
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
    
    async def run(self, start_url, max_depth=2):
        """Execute scraping"""
        try:
            proxy_status = "🌐 AUTO PROXY" if self.use_proxy else "🚫 NO PROXY"
            print(f"🚀 Scraping with Enhanced OCR {proxy_status}")
            print(f"🎯 Initial URL: {start_url}")
            print(f"📊 Maximum depth: {max_depth}")
            print(f"💾 Data will be saved to: {self.output_dir}")
            
            if self.use_proxy:
                # Wait for proxy configuration
                await self.setup_best_proxy()
                if self.current_proxy:
                    print(f"🌐 Active proxy: {self.current_proxy}")
            
            print("📸 Mode: Screenshot + Enhanced Image + OCR")
            
            await self.scrape_url(start_url, max_depth)
            
            # Close executor
            self.executor.shutdown(wait=True)
            
            print(f"\n✅ Scraping completed!")
            print(f"📄 Total pages processed: {len(self.visited_urls)}")
            print(f"📸 Original screenshots: {self.screenshots_dir}")
            print(f"🔧 Enhanced images: {self.enhanced_dir}")
            print(f"📝 Extracted texts: {self.texts_dir}")
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            self.executor.shutdown(wait=False)

async def main():
    """Main function"""
    url = input("Enter initial URL: ").strip()
    if not url:
        url = "https://example.com"
        print(f"Using default URL: {url}")
    
    depth = input("Enter maximum depth (default 2): ").strip()
    try:
        max_depth = int(depth) if depth else 2
    except ValueError:
        max_depth = 2
    
    # Ask about proxy
    use_proxy_input = input("Use corporate proxy? (Y/n): ").strip().lower()
    use_proxy = use_proxy_input not in ['n', 'no']
    
    scraper = ScreenshotScraper(use_proxy=use_proxy)
    await scraper.run(url, max_depth=max_depth)

if __name__ == "__main__":
    asyncio.run(main())