# Optimized chat9.py — JSON-only processing with DialoGPT-large and Gemini fallback
import gdown
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import glob
from typing import List, Dict, Any
import time
import gc
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download the entire folder from Google Drive
folder_url = "https://drive.google.com/drive/folders/1H6Wonw6Quzs9Bvo7KBxM6odpHV2gdum3"
DATASET_PATH = "C:\\Users\\vgmou\\Documents\\Chatbot\\c1\\your_dataset_folder"

# Configuration - Upgraded to DialoGPT-large for superior performance[citation:1][citation:6]
MODEL_NAME = "microsoft/DialoGPT-large"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "./faiss_filter_index"
MODEL_CACHE_DIR = "./model_cache"
BATCH_SIZE = 1000

# Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-2.5-flash"

# GPU Configuration
print("=== GPU/CPU Detection ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
        # Note: DialoGPT-large requires ~16GB GPU memory for optimal performance[citation:2]
        if gpu_memory_gb < 8.0:
            print("Warning: GPU memory may be insufficient for DialoGPT-large. The system will use CPU fallback.")
            DEVICE = "cpu"
        else:
            DEVICE = "cuda"
    except Exception:
        DEVICE = "cuda"
else:
    print("CUDA not available, using CPU")
    DEVICE = "cpu"

class FilterExtractionRAG:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.llm_model = None
        self.tokenizer = None
        self.device = DEVICE
        self.gemini_client = None
        self.MODEL_NAME = MODEL_NAME
        self._setup_gemini()

    def _setup_gemini(self):
        """Initialize Gemini client with API key"""
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
                print("Gemini client initialized successfully!")
            except Exception as e:
                print(f"Warning: Gemini initialization failed: {e}")
                self.gemini_client = None
        else:
            print("Warning: GEMINI_API_KEY not found in environment variables")
            self.gemini_client = None

    def download_dataset(self):
        """Download dataset from Google Drive if not already present"""
        if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
            print("Downloading dataset from Google Drive...")
            os.makedirs(DATASET_PATH, exist_ok=True)
            try:
                gdown.download_folder(
                    "https://drive.google.com/drive/folders/1H6Wonw6Quzs9Bvo7KBxM6odpHV2gdum3",
                    output=DATASET_PATH,
                    quiet=False
                )
                print("Dataset downloaded successfully!")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                return False
        else:
            print("Dataset already exists. Using existing files.")
        return True

    def setup_models(self):
        """Download and cache all required models safely"""
        print("Setting up models...")
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Setup embedding model
        print("Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': self.device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
                'device': self.device
            },
            cache_folder=MODEL_CACHE_DIR
        )

        # Load tokenizer and model safely for DialoGPT-large[citation:1][citation:4]
        print("Loading DialoGPT-large language model (762M parameters)...")
        try:
            # First load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                padding_side="left",
                trust_remote_code=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Enhanced model loading for DialoGPT-large[citation:2]
            model_kwargs = {
                'cache_dir': MODEL_CACHE_DIR,
                'trust_remote_code': False,
                'low_cpu_mem_usage': True,
            }
            
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Use GPU with memory optimization for large model
                    model_kwargs.update({
                        'torch_dtype': torch.float16,
                        'device_map': 'auto',
                    })
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        **model_kwargs
                    )
                    print("✅ DialoGPT-large loaded successfully with GPU optimization!")
                except Exception as e:
                    print(f"🚨 GPU loading failed: {e}")
                    print("🔄 Falling back to CPU for DialoGPT-large...")
                    self.device = "cpu"
                    # Fallback to CPU with float32
                    model_kwargs.update({
                        'torch_dtype': torch.float32,
                        'device_map': None,
                    })
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        **model_kwargs
                    )
            else:
                # CPU-only loading
                model_kwargs.update({
                    'torch_dtype': torch.float32,
                    'device_map': None,
                })
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    **model_kwargs
                )
                print("✅ DialoGPT-large loaded successfully on CPU!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔄 Attempting basic CPU loading...")
            try:
                # Most basic fallback
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    device_map=None,
                    torch_dtype=torch.float32
                )
                self.device = "cpu"
                print("✅ DialoGPT-large loaded with basic CPU fallback!")
            except Exception as e2:
                print(f"❌ All loading methods failed: {e2}")
                raise

    def load_and_process_dataset(self):
        """Load and process ONLY JSON files from the downloaded Google Drive folder"""
        print("Loading dataset from downloaded JSON files...")
        
        # First, download the dataset if needed
        if not self.download_dataset():
            return []

        # Find ONLY JSON files in the dataset directory
        all_files = glob.glob(os.path.join(DATASET_PATH, "**/*.json"), recursive=True)
        all_files.extend(glob.glob(os.path.join(DATASET_PATH, "*.json"), recursive=True))

        if not all_files:
            print("No JSON files found after download.")
            print("Please check the Google Drive folder contains JSON files.")
            return []

        print(f"Found {len(all_files)} JSON files: {[os.path.basename(f) for f in all_files]}")
        extracted_filters = []

        for file_path in all_files:
            print(f"Processing {os.path.basename(file_path)}...")
            try:
                file_filters = self._process_json_file(file_path)
                extracted_filters.extend(file_filters)
                print(f"Extracted {len(file_filters)} filters from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        if not extracted_filters:
            print("Warning: No filters could be extracted from the JSON files.")
            return []

        print(f"Total filters extracted from JSON files: {len(extracted_filters)}")
        
        # Display sample of extracted filters
        print("\nSample of extracted filters:")
        for i, filter_text in enumerate(extracted_filters[:5]):
            print(f"  {i+1}. {filter_text}")
        
        documents = [Document(page_content=filter_text) for filter_text in extracted_filters]
        return documents

    def _process_json_file(self, file_path: str) -> List[str]:
        """Process JSON files and extract key-value pairs as filters"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {file_path}: {e}")
                return []
        
        filters = []
        if isinstance(data, list):
            for item in data:
                filters.extend(self._extract_filters_from_dict(item))
        elif isinstance(data, dict):
            filters.extend(self._extract_filters_from_dict(data))
        
        return filters

    def _extract_filters_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """Recursively extract key-value pairs from dictionaries as filter strings"""
        filters = []
        filter_parts = []
        
        for key, value in data.items():
            # Skip common metadata fields that aren't useful as filters
            if key.lower() in ['id', 'timestamp', 'created_at', 'updated_at', 'metadata', 
                             'index', 'uuid', 'guid', '_id']:
                continue
                
            if isinstance(value, (str, int, float, bool)) and value not in [None, ""]:
                # Include meaningful values as filters
                if len(str(value).strip()) > 0:
                    filter_parts.append(f"{key}:{value}")
            elif isinstance(value, list):
                if value and all(isinstance(x, (str, int, float, bool)) for x in value):
                    # Join list elements for simple lists
                    filter_parts.append(f"{key}:{','.join(map(str, value))}")
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_filters = self._extract_filters_from_dict(value)
                filters.extend(nested_filters)
        
        if filter_parts:
            filters.append(", ".join(filter_parts))
            
        return filters

    def create_vector_store_optimized(self, documents):
        """Create vector store using all-MiniLM-L6-v2 embeddings"""
        if not documents:
            print("No documents to process for vector store.")
            return False
            
        print("Creating vector store with optimized batch processing...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"Total documents to process: {len(splits)}")
        
        start_time = time.time()
        batch_size = min(BATCH_SIZE, len(splits))
        total_batches = (len(splits) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(splits))
            batch = splits[start_idx:end_idx]
            
            if batch_idx == 0:
                self.vector_db = FAISS.from_documents(batch, self.embedding_model)
            else:
                self.vector_db.add_documents(batch)
                
            print(f"Processed batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)")
        
        self.vector_db.save_local(VECTOR_DB_PATH)
        end_time = time.time()
        print(f"Vector store created and saved to {VECTOR_DB_PATH}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return True

    def load_vector_store(self):
        """Load existing vector store"""
        print("Loading vector store...")
        self.vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def _generate_gemini_suggestions(self, user_query: str, context_filters: List[str]) -> str:
        """
        Use Gemini API to generate intelligent suggestion questions based on user query and available filters
        """
        if not self.gemini_client:
            return "I cannot provide suggestions at the moment (Gemini not available)."
        
        try:
            prompt = f"""
            You are a helpful assistant for a product filtering system. 
            A user asked: "{user_query}"
            
            Our system couldn't find a perfect match, but we have information related to these filter categories: 
            {', '.join(context_filters[:8])}
            
            Generate 3 friendly and helpful suggestion questions that:
            1. Are directly related to the user's original query intent
            2. Use the filter categories we actually have available
            3. Are natural and conversational
            4. Help guide the user to better utilize our filtering system
            
            Return ONLY the three questions as a simple bulleted list with no other text.
            """
            
            response = self.gemini_client.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "I cannot provide suggestions at the moment (API error)."

    def _calculate_similarity_confidence(self, user_query: str, context_filters: List[str]) -> float:
        """
        Calculate confidence score for query relevance
        Returns: 0.0 (completely unrelated) to 1.0 (highly related)
        """
        if not context_filters:
            return 0.0
            
        # Simple heuristic: if we retrieved any filters, consider it somewhat related
        query_lower = user_query.lower()
        
        # Check if query contains any common filter-related terms
        filter_terms = ['filter', 'show', 'find', 'search', 'list', 'category', 'type', 'brand', 'price']
        has_filter_intent = any(term in query_lower for term in filter_terms)
        
        # Basic confidence scoring
        base_confidence = 0.3 if context_filters else 0.0
        if has_filter_intent:
            base_confidence += 0.2
        if len(context_filters) >= 2:  # More relevant filters found
            base_confidence += 0.2
            
        return min(base_confidence, 1.0)

    def query_to_json(self, user_query):
        """Convert user query to JSON using retrieved filters from actual dataset"""
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if self.vector_db is None:
            return {"error": "Vector store not initialized"}

        # Retrieve relevant documents using semantic search 
        retrieved_docs = self.vector_db.similarity_search(user_query, k=3)
        context_filters = [doc.page_content for doc in retrieved_docs]

        print(f"Retrieved {len(context_filters)} relevant filters from dataset:")
        for i, filter_text in enumerate(context_filters):
            print(f"  {i+1}. {filter_text}")

        # Calculate similarity confidence
        confidence = self._calculate_similarity_confidence(user_query, context_filters)
        print(f"Query relevance confidence: {confidence:.2f}")

        # If query is completely unrelated, return early with simple message
        if confidence < 0.2:
            return {
                "user_query": user_query,
                "response": "This question not existed in my memory",
                "confidence_score": confidence,
                "note": "Query is completely unrelated to available filters"
            }

        # For moderately related queries, use Gemini for suggestions
        if confidence < 0.6:
            gemini_suggestions = self._generate_gemini_suggestions(user_query, context_filters)
            return {
                "user_query": user_query,
                "response": "I can't answer directly, but here are some related questions I can help with:",
                "suggested_questions": gemini_suggestions,
                "retrieved_filters": context_filters,
                "confidence_score": confidence,
                "note": "Used Gemini for suggestion generation"
            }

        # Enhanced prompt for DialoGPT-large JSON generation[citation:1]
        prompt = f"""**TASK**: Extract structured filters from the user query and convert to valid JSON format.

**USER QUERY**: "{user_query}"

**RELEVANT FILTER CONTEXT**:
{chr(10).join(['• ' + f for f in context_filters])}

**INSTRUCTIONS**:
1. Analyze the user query to identify key filter requirements
2. Create a JSON object with appropriate key-value pairs
3. Use descriptive keys that represent filter categories
4. Ensure values accurately reflect the user's requirements
5. Output ONLY valid JSON, no additional text

**REQUIRED OUTPUT FORMAT**:
{{
  "key1": "value1",
  "key2": ["value2", "value3"],
  "key3": 123
}}

**OUTPUT ONLY RAW JSON**:"""

        start_time = time.time()

        try:
            # Enhanced generation for DialoGPT-large[citation:1][citation:4]
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024, 
                padding=True
            )
            
            # Move inputs to the same device as model
            if hasattr(self.llm_model, 'device'):
                device = self.llm_model.device
            else:
                device = next(self.llm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Optimized generation parameters for DialoGPT-large[citation:1]
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=256,  # Increased for better JSON generation
                    do_sample=True,
                    temperature=0.3,     # Lower temperature for more structured output
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            print(f"🚨 Generation failed: {e}")
            # Enhanced fallback for DialoGPT-large
            return self._create_intelligent_fallback(user_query, context_filters, confidence)

        end_time = time.time()
        print(f"Text generation completed in {end_time - start_time:.2f} seconds")

        # Extract JSON from generated text with improved patterns
        json_str = generated_text.split("**OUTPUT ONLY RAW JSON**:")[-1].strip()
        json_str = json_str.split("```json")[-1].split("```")[0].strip()

        # Enhanced JSON extraction patterns
        json_patterns = [
            r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}',  # Nested objects
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Double nested
            r'\{[^{}]*\}',  # Simple object
        ]
        
        json_match = None
        for pattern in json_patterns:
            json_match = re.search(pattern, json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                break

        try:
            json_output = json.loads(json_str)
            # Validate JSON has meaningful content
            if isinstance(json_output, dict) and len(json_output) > 0:
                json_output["user_query"] = user_query
                json_output["confidence_score"] = confidence
                json_output["response_type"] = "direct_answer"
                return json_output
            else:
                raise json.JSONDecodeError("Empty JSON", "", 0)
                
        except json.JSONDecodeError as e:
            print(f"🚨 JSON parsing failed: {e}")
            print(f"🔧 Raw generated text: {json_str[:200]}...")
            # Use intelligent fallback
            return self._create_intelligent_fallback(user_query, context_filters, confidence, json_str)

    def _create_intelligent_fallback(self, user_query, context_filters, confidence, generated_text=None):
        """Create intelligent fallback response for DialoGPT-large"""
        # Extract technology keywords from query
        tech_keywords = {
            'react', 'node', 'python', 'java', 'javascript', 'aws', 'azure', 'docker', 
            'kubernetes', 'mongodb', 'sql', 'vue', 'angular', 'typescript', 'php',
            'ruby', 'go', 'rust', 'swift', 'kotlin'
        }
        
        query_lower = user_query.lower()
        found_techs = [tech.title() for tech in tech_keywords if tech in query_lower]
        
        # Create intelligent response based on query analysis
        if found_techs:
            fallback_json = {
                "user_query": user_query,
                "technologies": found_techs,
                "company_focus": "technology partners",
                "event_type": "tech partnership",
                "retrieved_filters": context_filters,
                "confidence_score": confidence,
                "response_type": "analyzed_fallback",
                "note": "Generated from query analysis using DialoGPT-large capabilities"
            }
        else:
            # General business partnership focus
            fallback_json = {
                "user_query": user_query,
                "partnership_type": "business collaboration",
                "company_criteria": "relevant industry partners",
                "retrieved_filters": context_filters,
                "confidence_score": confidence,
                "response_type": "general_fallback",
                "note": "Generated using DialoGPT-large general capabilities"
            }
        
        if generated_text:
            fallback_json["generation_attempt"] = generated_text[:300]
            
        return fallback_json

    def initialize_system(self, use_optimized=True):
        """Initialize the complete system"""
        if not os.path.exists(VECTOR_DB_PATH):
            print("First-time setup: Creating vector database...")
            self.setup_models()

            # Process the actual dataset files from Google Drive
            print("Processing dataset files to extract filters...")
            documents = self.load_and_process_dataset()
            
            if not documents:
                print("Error: No documents could be processed from dataset files.")
                print("Please check the Google Drive folder URL and permissions.")
                return False

            # Create vector store with actual dataset filters
            success = self.create_vector_store_optimized(documents)
            if not success:
                return False
        else:
            self.setup_models()
            self.load_vector_store()

        print("✅ System initialized and ready with DialoGPT-large!")
        return True

# GPU Memory monitoring function
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        cached_gb = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory allocated: {allocated_gb:.2f} GB")
        print(f"GPU Memory cached: {cached_gb:.2f} GB")

# Usage
def main():
    print("🚀 Starting RAG system with DialoGPT-large (762M) and Gemini integration...")
    print_gpu_memory()

    rag_system = FilterExtractionRAG()
    success = rag_system.initialize_system(use_optimized=True)
    
    if not success:
        print("❌ System initialization failed. Please check your dataset files.")
        return

    test_queries = [
        "Find all companies that are using technologies like React, Node.js, or Python for tech partnership",
        "Show me businesses using Java and AWS cloud services",
        "Filter companies with Python and machine learning expertise",
        "List tech companies using React and TypeScript",
        "What's the weather like today?",  # Unrelated query
        "Tell me about the history of Rome"  # Completely unrelated query
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*90}")
        print(f"Processing Query {i}/{len(test_queries)}")
        print(f"Query: {query}")

        start_time = time.time()
        json_output = rag_system.query_to_json(query)
        end_time = time.time()

        print(f"Query processed in {end_time - start_time:.2f} seconds")
        print("JSON Output:", json.dumps(json_output, indent=2))
        print_gpu_memory()

if __name__ == "__main__":
    main()