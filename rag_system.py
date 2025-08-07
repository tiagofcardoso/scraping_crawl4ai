import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiofiles
from datetime import datetime
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd

class RAGSystem:
    """RAG system for scraped and cleaned data"""
    
    def __init__(self, data_dir="scraped_data", model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.cleaned_dir = os.path.join(data_dir, "cleaned")
        self.rag_dir = os.path.join(data_dir, "rag")
        self.embeddings_dir = os.path.join(self.rag_dir, "embeddings")
        self.index_dir = os.path.join(self.rag_dir, "indexes")
        
        # Create directories
        os.makedirs(self.rag_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize embedding model
        print(f"🤖 Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Storage for documents and metadata
        self.documents = []
        self.metadata = []
        self.vector_store = None
        
        # Initialize OpenAI (optional - for advanced generation)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            print("✅ OpenAI API configured")
        else:
            print("⚠️  OpenAI API key not found - will use basic responses")
    
    async def load_cleaned_data(self) -> List[Dict]:
        """Load all cleaned JSON files"""
        try:
            json_files = list(Path(self.cleaned_dir).glob("*_cleaned.json"))
            
            if not json_files:
                print(f"❌ No cleaned JSON files found in {self.cleaned_dir}")
                return []
            
            print(f"📚 Loading {len(json_files)} cleaned files...")
            
            all_data = []
            for json_file in json_files:
                try:
                    async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                        data = json.loads(await f.read())
                        all_data.append(data)
                except Exception as e:
                    print(f"❌ Error loading {json_file}: {e}")
            
            print(f"✅ Loaded {len(all_data)} documents")
            return all_data
            
        except Exception as e:
            print(f"❌ Error loading cleaned data: {e}")
            return []
    
    async def create_document_chunks(self, cleaned_data: List[Dict]) -> List[Document]:
        """Create document chunks for embedding"""
        try:
            print(f"📄 Creating document chunks...")
            
            all_documents = []
            
            for data in cleaned_data:
                # Get text content
                full_text = data['content']['full_text']
                sentences = data['content']['sentences']
                
                # Create chunks from full text
                text_chunks = self.text_splitter.split_text(full_text)
                
                # Create documents with metadata
                for i, chunk in enumerate(text_chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    metadata = {
                        'source_file': data['processing']['file_source'],
                        'url': data['metadata'].get('url', 'Unknown'),
                        'language': data['content']['language'],
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'keywords': [kw[0] for kw in data['content']['keywords'][:5]],
                        'cleaned_at': data['processing']['cleaned_at'],
                        'chunk_type': 'full_text'
                    }
                    
                    doc = Document(page_content=chunk, metadata=metadata)
                    all_documents.append(doc)
                
                # Also add meaningful sentences as separate documents
                for i, sentence in enumerate(sentences[:20]):  # Limit to top 20 sentences
                    if len(sentence.strip()) < 30:
                        continue
                    
                    metadata = {
                        'source_file': data['processing']['file_source'],
                        'url': data['metadata'].get('url', 'Unknown'),
                        'language': data['content']['language'],
                        'sentence_index': i,
                        'keywords': [kw[0] for kw in data['content']['keywords'][:3]],
                        'cleaned_at': data['processing']['cleaned_at'],
                        'chunk_type': 'sentence'
                    }
                    
                    doc = Document(page_content=sentence, metadata=metadata)
                    all_documents.append(doc)
            
            print(f"✅ Created {len(all_documents)} document chunks")
            return all_documents
            
        except Exception as e:
            print(f"❌ Error creating document chunks: {e}")
            return []
    
    async def build_vector_index(self, documents: List[Document]) -> bool:
        """Build FAISS vector index"""
        try:
            print(f"🔍 Building vector index for {len(documents)} documents...")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            vector_store_path = os.path.join(self.index_dir, "faiss_index")
            self.vector_store.save_local(vector_store_path)
            
            # Save document metadata
            metadata_path = os.path.join(self.index_dir, "document_metadata.json")
            doc_metadata = []
            for doc in documents:
                doc_metadata.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(doc_metadata, ensure_ascii=False, indent=2))
            
            print(f"✅ Vector index built and saved to {vector_store_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error building vector index: {e}")
            return False
    
    async def load_vector_index(self) -> bool:
        """Load existing vector index"""
        try:
            vector_store_path = os.path.join(self.index_dir, "faiss_index")
            
            if not os.path.exists(vector_store_path):
                print(f"❌ No vector index found at {vector_store_path}")
                return False
            
            print(f"📚 Loading vector index from {vector_store_path}")
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
            
            print(f"✅ Vector index loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector index: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized")
                return []
            
            # Search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    async def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """Generate response using context documents"""
        try:
            # Combine context
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            if self.openai_api_key:
                # Use OpenAI for advanced generation
                response = await self.generate_openai_response(query, context_text)
            else:
                # Use basic template response
                response = self.generate_basic_response(query, context_text, context_docs)
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Desculpe, ocorreu um erro ao gerar a resposta: {e}"
    
    async def generate_openai_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI"""
        try:
            prompt = f"""
Baseado no contexto fornecido abaixo, responda à pergunta de forma precisa e informativa.
Use apenas as informações do contexto fornecido.

CONTEXTO:
{context}

PERGUNTA: {query}

RESPOSTA:
"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em analisar dados extraídos de websites. Responda sempre em português de forma clara e precisa."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error with OpenAI: {e}")
            return self.generate_basic_response(query, context, [])
    
    def generate_basic_response(self, query: str, context: str, docs: List[Document]) -> str:
        """Generate basic template response"""
        try:
            # Extract sources
            sources = set()
            languages = set()
            
            for doc in docs:
                if hasattr(doc, 'metadata'):
                    url = doc.metadata.get('url', 'Unknown')
                    if url != 'Unknown':
                        sources.add(url)
                    languages.add(doc.metadata.get('language', 'Unknown'))
            
            # Create basic response
            response = f"""
Com base nos dados extraídos dos websites, encontrei as seguintes informações relacionadas à sua pergunta "{query}":

{context[:1000]}{"..." if len(context) > 1000 else ""}

Fontes consultadas:
"""
            
            for i, source in enumerate(list(sources)[:3], 1):
                response += f"{i}. {source}\n"
            
            if len(sources) > 3:
                response += f"... e mais {len(sources) - 3} fontes\n"
            
            response += f"\nIdiomas detectados: {', '.join(languages)}"
            response += f"\nDocumentos analisados: {len(docs)}"
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating basic response: {e}")
            return "Desculpe, não foi possível gerar uma resposta adequada."
    
    async def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Main query function"""
        try:
            print(f"🔍 Searching for: {question}")
            
            # Search similar documents
            results = self.search_similar_documents(question, k=k)
            
            if not results:
                return {
                    'question': question,
                    'answer': 'Não foram encontrados documentos relevantes para esta pergunta.',
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Extract documents and scores
            docs = [result[0] for result in results]
            scores = [result[1] for result in results]
            
            # Generate response
            answer = await self.generate_response(question, docs)
            
            # Prepare sources
            sources = []
            for doc, score in results:
                source_info = {
                    'url': doc.metadata.get('url', 'Unknown'),
                    'file': doc.metadata.get('source_file', 'Unknown'),
                    'language': doc.metadata.get('language', 'Unknown'),
                    'relevance_score': float(score),
                    'keywords': doc.metadata.get('keywords', []),
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            # Calculate average confidence
            avg_confidence = 1.0 - (sum(scores) / len(scores)) if scores else 0.0
            
            result = {
                'question': question,
                'answer': answer,
                'sources': sources,
                'confidence': avg_confidence,
                'timestamp': datetime.now().isoformat(),
                'documents_found': len(docs)
            }
            
            print(f"✅ Generated response with {len(docs)} sources")
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                'question': question,
                'answer': f'Erro ao processar a pergunta: {e}',
                'sources': [],
                'confidence': 0.0
            }
    
    async def build_rag_index(self) -> bool:
        """Build complete RAG index from cleaned data"""
        try:
            print("🚀 Building RAG index...")
            
            # Load cleaned data
            cleaned_data = await self.load_cleaned_data()
            if not cleaned_data:
                return False
            
            # Create document chunks
            documents = await self.create_document_chunks(cleaned_data)
            if not documents:
                return False
            
            # Build vector index
            success = await self.build_vector_index(documents)
            
            if success:
                print("✅ RAG index built successfully!")
                
                # Save index info
                index_info = {
                    'created_at': datetime.now().isoformat(),
                    'total_documents': len(documents),
                    'total_sources': len(cleaned_data),
                    'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
                
                info_path = os.path.join(self.rag_dir, "index_info.json")
                async with aiofiles.open(info_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(index_info, ensure_ascii=False, indent=2))
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error building RAG index: {e}")
            return False

async def main():
    """Main function for RAG system testing"""
    print("🤖 RAG System for Scraped Data")
    print("=" * 50)
    
    # Initialize RAG system
    data_dir = input("Enter scraped data directory (default: scraped_data): ").strip()
    if not data_dir:
        data_dir = "scraped_data"
    
    rag = RAGSystem(data_dir)
    
    # Check if index exists
    index_exists = await rag.load_vector_index()
    
    if not index_exists:
        print("🔧 No existing index found. Building new index...")
        success = await rag.build_rag_index()
        if not success:
            print("❌ Failed to build RAG index")
            return
    
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
            
            if result['sources']:
                print(f"\n🔗 Top sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"{i}. {source['url']} (relevance: {source['relevance_score']:.3f})")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())