import os
import json
import asyncio
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import aiofiles
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenAI with new API format
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system for document search and Q&A"""
    
    def __init__(self, data_dir="scraped_data"):
        self.data_dir = data_dir
        self.cleaned_dir = os.path.join(data_dir, "cleaned")
        self.rag_dir = os.path.join(data_dir, "rag")
        self.indexes_dir = os.path.join(self.rag_dir, "indexes")
        self.embeddings_dir = os.path.join(self.rag_dir, "embeddings")
        
        # Create directories
        os.makedirs(self.rag_dir, exist_ok=True)
        os.makedirs(self.indexes_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize models
        self.embedding_model = None
        self.openai_client = None
        self.faiss_index = None
        self.documents = []
        self.document_chunks = []
        
        # Configuration
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.top_k = 5

    async def initialize_models(self):
        """Initialize embedding model and OpenAI client"""
        try:
            # Initialize embedding model
            model_name = "all-MiniLM-L6-v2"
            print(f"🤖 Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            
            # Initialize OpenAI client with new API
            if OPENAI_AVAILABLE:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✅ OpenAI client initialized")
                else:
                    print("⚠️  OPENAI_API_KEY not found in environment variables")
            else:
                print("⚠️  OpenAI not available")
                
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise

    async def load_documents(self) -> List[Dict]:
        """Load cleaned documents from JSON files"""
        try:
            print("📚 Loading cleaned documents...")
            
            json_files = list(Path(self.cleaned_dir).glob("*_cleaned.json"))
            documents = []
            
            for json_file in json_files:
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                    # Extract relevant information
                    doc = {
                        'id': str(len(documents)),
                        'filename': data['processing']['file_source'],
                        'url': data['metadata'].get('url', 'Unknown'),
                        'content': data['content']['full_text'],
                        'language': data['content']['language'],
                        'keywords': [kw[0] for kw in data['content']['keywords'][:10]],
                        'metadata': data['metadata']
                    }
                    documents.append(doc)
            
            print(f"✅ Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return []

    def create_chunks(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into smaller chunks"""
        try:
            print("📄 Creating document chunks...")
            chunks = []
            
            for doc in documents:
                content = doc['content']
                words = content.split()
                
                # Create overlapping chunks
                for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    if len(chunk_text.strip()) > 50:  # Only keep meaningful chunks
                        chunk = {
                            'id': f"{doc['id']}_{len(chunks)}",
                            'text': chunk_text,
                            'doc_id': doc['id'],
                            'filename': doc['filename'],
                            'url': doc['url'],
                            'language': doc['language'],
                            'chunk_index': len(chunks)
                        }
                        chunks.append(chunk)
            
            print(f"✅ Created {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error creating chunks: {e}")
            return []

    async def build_vector_index(self, chunks: List[Dict]):
        """Build FAISS vector index from document chunks"""
        try:
            print(f"🔍 Building vector index for {len(chunks)} documents...")
            
            # Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Save index and chunks with proper file extension
            index_path = os.path.join(self.indexes_dir, "faiss_index.idx")
            faiss.write_index(self.faiss_index, index_path)
            
            chunks_path = os.path.join(self.embeddings_dir, "document_chunks.json")
            async with aiofiles.open(chunks_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(chunks, ensure_ascii=False, indent=2))
            
            self.document_chunks = chunks
            print(f"✅ Vector index built and saved to {index_path}")
            
        except Exception as e:
            print(f"❌ Error building vector index: {e}")
            raise

    async def load_existing_index(self) -> bool:
        """Load existing FAISS index and document chunks"""
        try:
            index_path = os.path.join(self.indexes_dir, "faiss_index.idx")
            chunks_path = os.path.join(self.embeddings_dir, "document_chunks.json")
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                # Initialize models first - this was missing!
                await self.initialize_models()
                
                # Load FAISS index
                self.faiss_index = faiss.read_index(index_path)
                
                # Load document chunks
                async with aiofiles.open(chunks_path, 'r', encoding='utf-8') as f:
                    self.document_chunks = json.loads(await f.read())
                
                print(f"✅ Loaded existing index with {len(self.document_chunks)} chunks")
                print(f"✅ Embedding model: {type(self.embedding_model).__name__}")
                print(f"✅ OpenAI client: {'Available' if self.openai_client else 'Not configured'}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error loading existing index: {e}")
            return False

    async def build_rag_index(self) -> bool:
        """Build complete RAG index"""
        try:
            print("🚀 Building RAG index...")
            
            # Initialize models
            await self.initialize_models()
            
            # Clean up any existing faiss_index directory if it exists
            old_index_dir = os.path.join(self.indexes_dir, "faiss_index")
            if os.path.isdir(old_index_dir):
                import shutil
                print("🧹 Removing old index directory...")
                shutil.rmtree(old_index_dir)
            
            # Try to load existing index
            if await self.load_existing_index():
                return True
            
            # Load documents
            self.documents = await self.load_documents()
            if not self.documents:
                print("❌ No documents found")
                return False
            
            # Create chunks
            chunks = self.create_chunks(self.documents)
            if not chunks:
                print("❌ No chunks created")
                return False
            
            # Build vector index
            await self.build_vector_index(chunks)
            
            print("✅ RAG index built successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error building RAG index: {e}")
            return False

    async def search_similar_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for similar documents using vector similarity"""
        try:
            if not self.faiss_index or not self.document_chunks:
                print("❌ RAG index not loaded")
                return []
            
            if not self.embedding_model:
                print("❌ Embedding model not initialized")
                print("🔧 Trying to initialize model...")
                model_name = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(model_name)
                print(f"✅ Embedding model initialized: {model_name}")
            
            if top_k is None:
                top_k = self.top_k
            
            print(f"🔍 Generating embeddings for query: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            print(f"✅ Query embedding shape: {query_embedding.shape}")
            
            # Search similar documents
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            print(f"🔍 Found {len(scores[0])} results with scores: {scores[0]}")
            
            # Retrieve matching chunks
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_chunks) and idx >= 0:
                    chunk = self.document_chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['rank'] = i + 1
                    results.append(chunk)
                    print(f"   📄 Result {i+1}: {chunk['filename']} (score: {score:.3f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer using OpenAI with new API format"""
        try:
            if not self.openai_client:
                return {
                    'answer': "OpenAI client not available. Please check your API key.",
                    'confidence': 0.0,
                    'error': 'No OpenAI client'
                }
            
            # Prepare context from similar chunks
            context = "\n\n".join([
                f"Document: {chunk['filename']}\nContent: {chunk['text'][:500]}..."
                for chunk in context_chunks[:3]
            ])
            
            # Create prompt
            prompt = f"""Based on the following context from scraped documents, answer the question.
If the information is not available in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
            
            # Use new OpenAI API format
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from scraped documents. Also answer any questions from the user."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in context_chunks]) if context_chunks else 0
            confidence = min(avg_similarity * 1.2, 1.0)  # Scale to 0-1
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': [chunk['filename'] for chunk in context_chunks],
                'context_used': len(context_chunks)
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'error': str(e)
            }

    async def query(self, question: str) -> Dict:
        """Main query function - search and generate answer"""
        try:
            print(f"🔍 Searching for: {question}")
            
            # Search for similar documents
            similar_docs = await self.search_similar_documents(question, top_k=5)
            
            if not similar_docs:
                return {
                    'answer': "No relevant documents found for your question.",
                    'confidence': 0.0,
                    'documents_found': 0,
                    'sources': []
                }
            
            print(f"📚 Found {len(similar_docs)} relevant document(s)")
            
            # Generate answer using OpenAI
            result = await self.generate_answer(question, similar_docs)
            
            # Add metadata
            result['documents_found'] = len(similar_docs)
            result['query'] = question
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'confidence': 0.0,
                'documents_found': 0,
                'sources': [],
                'error': str(e)
            }

async def main():
    """Main function for RAG system testing"""
    print("🤖 RAG System for Scraped Data")
    print("=" * 50)
    
    # Initialize RAG system
    data_dir = input("Enter scraped data directory (default: scraped_data): ").strip()
    if not data_dir:
        data_dir = "scraped_data"
    
    rag = RAGSystem(data_dir)
    
    # Check if index exists and load models
    index_exists = await rag.load_existing_index()
    
    if not index_exists:
        print("🔧 No existing index found. Building new index...")
        success = await rag.build_rag_index()
        if not success:
            print("❌ Failed to build RAG index")
            return
    
    # Verify all components are loaded
    print(f"\n🔍 System Status:")
    print(f"   FAISS Index: {'✅ Loaded' if rag.faiss_index else '❌ Missing'}")
    print(f"   Document Chunks: {'✅ Loaded' if rag.document_chunks else '❌ Missing'} ({len(rag.document_chunks) if rag.document_chunks else 0} chunks)")
    print(f"   Embedding Model: {'✅ Ready' if rag.embedding_model else '❌ Missing'}")
    print(f"   OpenAI Client: {'✅ Ready' if rag.openai_client else '❌ Missing'}")
    
    # Interactive query loop
    print("\n🎯 RAG System ready! Ask questions about your scraped data.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Process query
            result = await rag.query(question)
            
            print(f"\n💡 Answer:")
            print(f"{result['answer']}")
            print(f"\n📊 Confidence: {result['confidence']:.2f}")
            print(f"📚 Sources found: {result['documents_found']}")
            
            if result.get('sources') and isinstance(result['sources'], list) and len(result['sources']) > 0:
                print(f"\n🔗 Top sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    if isinstance(source, str):
                        print(f"{i}. {source}")
                    elif isinstance(source, dict) and 'filename' in source:
                        print(f"{i}. {source['filename']}")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())