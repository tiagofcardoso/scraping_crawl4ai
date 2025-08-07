import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import aiofiles
from datetime import datetime
from collections import Counter
import unicodedata

# Try to import NLTK with fallbacks
try:
    import nltk
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
    
except ImportError:
    print("⚠️  NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False
    
    # Fallback tokenizers
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)

class DataCleaner:
    """Advanced data cleaning for scraped content"""
    
    def __init__(self, scraped_data_dir="scraped_data"):
        self.scraped_data_dir = scraped_data_dir
        self.texts_dir = os.path.join(scraped_data_dir, "texts")
        self.cleaned_dir = os.path.join(scraped_data_dir, "cleaned")
        self.analytics_dir = os.path.join(scraped_data_dir, "analytics")
        
        # Create directories
        os.makedirs(self.cleaned_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Language configurations
        if NLTK_AVAILABLE:
            try:
                self.stop_words_pt = set(stopwords.words('portuguese'))
                self.stop_words_en = set(stopwords.words('english'))
            except:
                print("⚠️  NLTK stopwords not available, using basic list")
                self.stop_words_pt = self._get_basic_stopwords_pt()
                self.stop_words_en = self._get_basic_stopwords_en()
        else:
            print("⚠️  Using basic stopword lists (NLTK not available)")
            self.stop_words_pt = self._get_basic_stopwords_pt()
            self.stop_words_en = self._get_basic_stopwords_en()
        
        self.all_stop_words = self.stop_words_pt.union(self.stop_words_en)
        
        # Common noise patterns
        self.noise_patterns = [
            r'cookies?\s+policy',
            r'privacy\s+policy',
            r'terms?\s+of\s+service',
            r'sign\s+in',
            r'log\s+in',
            r'register',
            r'subscribe',
            r'newsletter',
            r'advertisement',
            r'loading\.+',
            r'click\s+here',
            r'read\s+more',
            r'continue\s+reading',
            r'©\s*\d{4}',
            r'all\s+rights?\s+reserved',
            r'powered\s+by',
            r'developed\s+by',
            r'designed\s+by'
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.noise_patterns]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text encoding and characters"""
        try:
            # Normalize unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Fix common encoding issues
            replacements = {
                'â€™': "'",
                'â€œ': '"',
                'â€': '"',
                'â€¢': '•',
                'â€"': '—',
                'â€"': '–',
                'Ã¡': 'á',
                'Ã©': 'é',
                'Ã­': 'í',
                'Ã³': 'ó',
                'Ãº': 'ú',
                'Ã§': 'ç',
                'Ã ': 'à',
                'Ãª': 'ê',
                'Ã´': 'ô',
                'Ã±': 'ñ'
            }
            
            for wrong, correct in replacements.items():
                text = text.replace(wrong, correct)
            
            return text
        except Exception as e:
            print(f"⚠️  Error normalizing text: {e}")
            return text
    
    def remove_ocr_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts and errors"""
        try:
            # Remove common OCR artifacts
            ocr_artifacts = [
                r'\|\s*\|\s*\|',  # Table artifacts
                r'[|]{2,}',       # Multiple pipes
                r'_{3,}',         # Multiple underscores
                r'-{3,}',         # Multiple dashes
                r'={3,}',         # Multiple equals
                r'\s+[.]{3,}\s+', # Multiple dots
                r'[^\w\s.,;:!?()\'\"àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇñÑ\-]', # Invalid characters
                r'\b[a-zA-Z]\b',  # Single letters (except 'a', 'e', 'o')
                r'(?<!\w)[a-zA-Z](?=\s)',  # Single characters at word boundaries
                r'\s[a-zA-Z]\s',  # Single letters between spaces
            ]
            
            for pattern in ocr_artifacts:
                text = re.sub(pattern, ' ', text)
            
            # Fix common OCR misreads
            ocr_fixes = {
                r'\brnm\b': 'mm',
                r'\brn\b': 'in',
                r'\bvvith\b': 'with',
                r'\bvvhen\b': 'when',
                r'\bvvhere\b': 'where',
                r'\bvvhat\b': 'what',
                r'\bteh\b': 'the',
                r'\badn\b': 'and',
                r'\bfrom\s+the\s+(?:top|bottom)\b': '',
                r'\b(?:page|pag)\s+\d+\b': '',
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b': '',  # Times
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b': '',  # Dates
            }
            
            for pattern, replacement in ocr_fixes.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            return text
        except Exception as e:
            print(f"⚠️  Error removing OCR artifacts: {e}")
            return text
    
    def remove_noise_content(self, text: str) -> str:
        """Remove noise like navigation, ads, etc."""
        try:
            # Remove noise patterns
            for pattern in self.compiled_patterns:
                text = pattern.sub('', text)
            
            # Remove common UI elements
            ui_patterns = [
                r'menu\s+toggle',
                r'skip\s+to\s+content',
                r'skip\s+to\s+main',
                r'breadcrumb',
                r'search\s+for:?',
                r'sort\s+by',
                r'filter\s+by',
                r'show\s+more',
                r'load\s+more',
                r'view\s+all',
                r'see\s+all',
                r'back\s+to\s+top',
                r'scroll\s+to\s+top',
            ]
            
            for pattern in ui_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            return text
        except Exception as e:
            print(f"⚠️  Error removing noise: {e}")
            return text
    
    def clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Remove empty lines
            lines = text.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            text = '\n'.join(lines)
            
            return text
        except Exception as e:
            print(f"⚠️  Error cleaning whitespace: {e}")
            return text
    
    def extract_meaningful_sentences(self, text: str, min_words=5, max_words=100) -> List[str]:
        """Extract meaningful sentences"""
        try:
            sentences = sent_tokenize(text)
            meaningful_sentences = []
            
            for sentence in sentences:
                # Clean sentence
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Count words
                words = word_tokenize(sentence.lower())
                word_count = len([w for w in words if w.isalpha()])
                
                # Filter by word count
                if min_words <= word_count <= max_words:
                    # Check if sentence has enough meaningful content
                    meaningful_words = [w for w in words if w.isalpha() and w not in self.all_stop_words]
                    
                    if len(meaningful_words) >= 3:  # At least 3 meaningful words
                        meaningful_sentences.append(sentence)
            
            return meaningful_sentences
        except Exception as e:
            print(f"⚠️  Error extracting sentences: {e}")
            return [text]  # Return original if fails
    
    def extract_keywords(self, text: str, top_n=20) -> List[tuple]:
        """Extract top keywords from text"""
        try:
            # Tokenize and clean
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and len(word) > 2]
            words = [word for word in words if word not in self.all_stop_words]
            
            # Count frequencies
            word_freq = Counter(words)
            
            return word_freq.most_common(top_n)
        except Exception as e:
            print(f"⚠️  Error extracting keywords: {e}")
            return []
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        try:
            # Count Portuguese vs English words
            words = word_tokenize(text.lower())
            pt_count = sum(1 for word in words if word in self.stop_words_pt)
            en_count = sum(1 for word in words if word in self.stop_words_en)
            
            if pt_count > en_count:
                return "Portuguese"
            elif en_count > pt_count:
                return "English"
            else:
                return "Mixed/Unknown"
        except Exception as e:
            print(f"⚠️  Error detecting language: {e}")
            return "Unknown"
    
    def parse_metadata(self, text: str) -> Dict[str, Any]:
        """Parse metadata from text file"""
        metadata = {}
        lines = text.split('\n')
        
        for line in lines[:20]:  # Check first 20 lines for metadata
            if ':' in line and '=' not in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key.lower() in ['url', 'date', 'depth', 'method', 'proxy', 'time', 'characters', 'words']:
                        metadata[key.lower()] = value
                except:
                    continue
        
        return metadata
    
    async def clean_text_file(self, file_path: str) -> Dict[str, Any]:
        """Clean a single text file"""
        try:
            print(f"🧹 Cleaning: {os.path.basename(file_path)}")
            
            # Read file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse metadata
            metadata = self.parse_metadata(content)
            
            # Extract main text (after metadata separator)
            if '=' * 20 in content:
                text_parts = content.split('=' * 20)
                main_text = text_parts[-1] if len(text_parts) > 1 else content
            else:
                main_text = content
            
            # Original stats
            original_chars = len(main_text)
            original_words = len(main_text.split())
            
            # Step-by-step cleaning
            print(f"  📝 Original: {original_chars} chars, {original_words} words")
            
            # 1. Normalize text
            cleaned_text = self.normalize_text(main_text)
            
            # 2. Remove OCR artifacts
            cleaned_text = self.remove_ocr_artifacts(cleaned_text)
            
            # 3. Remove noise content
            cleaned_text = self.remove_noise_content(cleaned_text)
            
            # 4. Clean whitespace
            cleaned_text = self.clean_whitespace(cleaned_text)
            
            # Final stats
            final_chars = len(cleaned_text)
            final_words = len(cleaned_text.split())
            
            print(f"  ✨ Cleaned: {final_chars} chars, {final_words} words")
            print(f"  📊 Reduction: {((original_chars - final_chars) / original_chars * 100):.1f}% chars")
            
            # Extract meaningful content
            sentences = self.extract_meaningful_sentences(cleaned_text)
            keywords = self.extract_keywords(cleaned_text)
            language = self.detect_language(cleaned_text)
            
            # Create cleaned data structure
            cleaned_data = {
                'metadata': metadata,
                'original_stats': {
                    'characters': original_chars,
                    'words': original_words
                },
                'cleaned_stats': {
                    'characters': final_chars,
                    'words': final_words,
                    'sentences': len(sentences),
                    'reduction_percent': ((original_chars - final_chars) / original_chars * 100) if original_chars > 0 else 0
                },
                'content': {
                    'full_text': cleaned_text,
                    'sentences': sentences,
                    'keywords': keywords,
                    'language': language
                },
                'processing': {
                    'cleaned_at': datetime.now().isoformat(),
                    'file_source': os.path.basename(file_path)
                }
            }
            
            # Save cleaned version
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Save as JSON
            json_path = os.path.join(self.cleaned_dir, f"{base_name}_cleaned.json")
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cleaned_data, ensure_ascii=False, indent=2))
            
            # Save as clean text
            txt_path = os.path.join(self.cleaned_dir, f"{base_name}_clean.txt")
            async with aiofiles.open(txt_path, 'w', encoding='utf-8') as f:
                await f.write(cleaned_text)
            
            print(f"  💾 Saved: {json_path}")
            print(f"  💾 Saved: {txt_path}")
            
            return cleaned_data
            
        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
            return {}
    
    async def clean_all_files(self) -> Dict[str, Any]:
        """Clean all text files in the texts directory"""
        try:
            text_files = list(Path(self.texts_dir).glob("*.txt"))
            
            if not text_files:
                print(f"⚠️  No text files found in {self.texts_dir}")
                return {}
            
            print(f"🧹 Starting to clean {len(text_files)} files...")
            
            all_cleaned_data = []
            stats = {
                'total_files': len(text_files),
                'processed_files': 0,
                'failed_files': 0,
                'total_original_chars': 0,
                'total_cleaned_chars': 0,
                'languages': Counter(),
                'all_keywords': Counter()
            }
            
            for file_path in text_files:
                try:
                    cleaned_data = await self.clean_text_file(str(file_path))
                    
                    if cleaned_data:
                        all_cleaned_data.append(cleaned_data)
                        stats['processed_files'] += 1
                        stats['total_original_chars'] += cleaned_data['original_stats']['characters']
                        stats['total_cleaned_chars'] += cleaned_data['cleaned_stats']['characters']
                        stats['languages'][cleaned_data['content']['language']] += 1
                        
                        # Aggregate keywords
                        for keyword, count in cleaned_data['content']['keywords']:
                            stats['all_keywords'][keyword] += count
                    else:
                        stats['failed_files'] += 1
                        
                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")
                    stats['failed_files'] += 1
            
            # Generate analytics
            analytics = await self.generate_analytics(all_cleaned_data, stats)
            
            # Save analytics
            analytics_path = os.path.join(self.analytics_dir, "cleaning_analytics.json")
            async with aiofiles.open(analytics_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(analytics, ensure_ascii=False, indent=2))
            
            print(f"\n✅ Cleaning completed!")
            print(f"📊 Processed: {stats['processed_files']}/{stats['total_files']} files")
            print(f"📊 Failed: {stats['failed_files']} files")
            print(f"📊 Total reduction: {((stats['total_original_chars'] - stats['total_cleaned_chars']) / stats['total_original_chars'] * 100):.1f}%" if stats['total_original_chars'] > 0 else "N/A")
            print(f"💾 Cleaned files: {self.cleaned_dir}")
            print(f"📈 Analytics: {analytics_path}")
            
            return analytics
            
        except Exception as e:
            print(f"❌ Error in batch cleaning: {e}")
            return {}
    
    async def generate_analytics(self, all_cleaned_data: List[Dict], stats: Dict) -> Dict[str, Any]:
        """Generate comprehensive analytics"""
        try:
            analytics = {
                'summary': {
                    'total_files': stats['total_files'],
                    'processed_files': stats['processed_files'],
                    'failed_files': stats['failed_files'],
                    'success_rate': (stats['processed_files'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0,
                    'total_original_characters': stats['total_original_chars'],
                    'total_cleaned_characters': stats['total_cleaned_chars'],
                    'overall_reduction_percent': ((stats['total_original_chars'] - stats['total_cleaned_chars']) / stats['total_original_chars'] * 100) if stats['total_original_chars'] > 0 else 0,
                    'processing_date': datetime.now().isoformat()
                },
                'language_distribution': dict(stats['languages']),
                'top_keywords': dict(stats['all_keywords'].most_common(50)),
                'file_details': []
            }
            
            # Add file details
            for data in all_cleaned_data:
                file_detail = {
                    'file': data['processing']['file_source'],
                    'url': data['metadata'].get('url', 'Unknown'),
                    'language': data['content']['language'],
                    'original_chars': data['original_stats']['characters'],
                    'cleaned_chars': data['cleaned_stats']['characters'],
                    'reduction_percent': data['cleaned_stats']['reduction_percent'],
                    'sentences': data['cleaned_stats']['sentences'],
                    'top_keywords': dict(data['content']['keywords'][:10])
                }
                analytics['file_details'].append(file_detail)
            
            return analytics
            
        except Exception as e:
            print(f"❌ Error generating analytics: {e}")
            return {}
    
    async def export_to_csv(self) -> str:
        """Export cleaned data to CSV for analysis"""
        try:
            # Load all cleaned JSON files
            json_files = list(Path(self.cleaned_dir).glob("*_cleaned.json"))
            
            if not json_files:
                print("⚠️  No cleaned JSON files found")
                return ""
            
            records = []
            
            for json_file in json_files:
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                
                record = {
                    'filename': data['processing']['file_source'],
                    'url': data['metadata'].get('url', ''),
                    'language': data['content']['language'],
                    'original_chars': data['original_stats']['characters'],
                    'cleaned_chars': data['cleaned_stats']['characters'],
                    'reduction_percent': data['cleaned_stats']['reduction_percent'],
                    'sentences_count': data['cleaned_stats']['sentences'],
                    'words_count': data['cleaned_stats']['words'],
                    'top_3_keywords': ', '.join([kw[0] for kw in data['content']['keywords'][:3]]),
                    'cleaned_at': data['processing']['cleaned_at']
                }
                records.append(record)
            
            # Create DataFrame and save
            df = pd.DataFrame(records)
            csv_path = os.path.join(self.analytics_dir, "cleaned_data_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"📊 CSV exported: {csv_path}")
            return csv_path
            
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return ""

async def main():
    """Main function for data cleaning"""
    print("🧹 Data Cleaning Tool")
    print("=" * 50)
    
    # Get data directory
    data_dir = input("Enter scraped data directory (default: scraped_data): ").strip()
    if not data_dir:
        data_dir = "scraped_data"
    
    if not os.path.exists(os.path.join(data_dir, "texts")):
        print(f"❌ No texts directory found in {data_dir}")
        return
    
    # Initialize cleaner
    cleaner = DataCleaner(data_dir)
    
    # Run cleaning
    analytics = await cleaner.clean_all_files()
    
    # Export CSV
    if analytics:
        csv_path = await cleaner.export_to_csv()
        
        print(f"\n📈 Cleaning Summary:")
        print(f"Files processed: {analytics['summary']['processed_files']}")
        print(f"Success rate: {analytics['summary']['success_rate']:.1f}%")
        print(f"Overall reduction: {analytics['summary']['overall_reduction_percent']:.1f}%")
        print(f"Languages found: {', '.join(analytics['language_distribution'].keys())}")
        print(f"Top keywords: {', '.join(list(analytics['top_keywords'].keys())[:10])}")

if __name__ == "__main__":
    asyncio.run(main())