# Bank RAG AI Agent 🏦

**Melbourne First Bank AI-Powered Customer Service Assistant**

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent Australian banking Q&A, built following enterprise best practices and PRD requirements.

> **🇦🇺 Australian Banking Content**: This system uses Melbourne First Bank as the example institution, with realistic Australian banking documents covering loans, credit cards, accounts, ATMs, overdraft protection, international transfers, and mobile banking.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Adding New Documents](#adding-new-documents)
- [Evaluation](#evaluation)
- [Production Monitoring & Optimization](#production-monitoring--optimization)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Bank RAG system provides intelligent, context-aware answers to banking questions using:

- **Hybrid Retrieval**: Combines BM25 (keyword) + FAISS (semantic) for optimal results
- **Open Source Models**: Ollama for LLM generation, sentence-transformers for embeddings
- **Citation Mechanism**: Tracks which documents were used to answer questions
- **Refusal Mechanism**: Declines to answer when confidence is low
- **Production-Ready**: Modular Python architecture ready for API integration

### Key Metrics (from PRD)
- ✅ Response time: < 3 seconds (target)
- ✅ Retrieval Precision: ≥ 90% (target)
- ✅ Generation Accuracy: ≥ 85% (target)

---

## ✨ Features

### Core Capabilities
- **Multi-format Support**: PDF, DOCX, HTML, TXT documents
- **Intelligent Chunking**: Optimized text splitting with overlap
- **Hybrid Search**: Best of keyword and semantic search
- **Source Attribution**: Every answer cites its sources
- **Confidence Scoring**: Know when the system is uncertain
- **Comprehensive Evaluation**: 9 metrics across retrieval, generation, and end-to-end quality (powered by Ragas)
- **Production Monitoring**: Automatic query logging, performance tracking, and optimization recommendations

### Australian Banking Content
- 7 comprehensive Melbourne First Bank documents (44,000+ words):
  - Personal loans policy
  - Credit card FAQ
  - Account types (transaction, savings, term deposits)
  - ATM & branch locations (Victoria-wide)
  - International money transfers (PayID, BPAY, Osko)
  - Mobile banking guide
  - Overdraft protection guide

---

## 🏗️ Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Hybrid Retriever          │
│  ┌──────────┬──────────┐    │
│  │   BM25   │  Vector  │    │
│  │ (Keyword)│(Semantic)│    │
│  └────┬─────┴─────┬────┘    │
│       └─────┬─────┘         │
│             │               │
│      ┌──────▼───────┐       │
│      │   Reranker   │       │
│      │  (Optional)  │       │
│      └──────┬───────┘       │
└─────────────┼───────────────┘
              │
              ▼
       ┌─────────────┐
       │   Context   │
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │  Ollama LLM │
       │  Generator  │
       └──────┬──────┘
              │
              ▼
┌──────────────────────────────┐
│  Response + Citations        │
│  + Confidence Score          │
└──────────────────────────────┘
```

### Technology Stack
- **LLM**: Ollama (Llama 3.1, Qwen, Mistral, etc.)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, multilingual-e5-base)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Keyword Search**: BM25
- **Framework**: LangChain
- **Language**: Python 3.8+

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running (see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md))
- **4GB+ RAM** (default lightweight config) or 8GB+ RAM (for better quality)
- See [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) for low-resource options

### Installation

```bash
# 1. Navigate to project directory
cd bank_RAG

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini  # Lightweight model (3.8B, ~4GB RAM) - default
# Or: ollama pull llama3.1:8b  # Better quality (8GB RAM)

# 5. Configure .env file (optional, defaults work)
# See MIGRATION_GUIDE.md for configuration options
```

### First Run

```bash
# Interactive mode (recommended for testing)
python main.py interactive

# Or batch mode
python main.py batch

# Or evaluation mode
python main.py eval
```

---

## 💻 Usage

### 1. Interactive Mode

```bash
python main.py interactive
```

Ask questions naturally:
```
💬 Your question: What are the fees for wire transfers?

🤖 Answer:
For TechBank customers, wire transfer fees vary by account type:
- Basic/Preferred Checking: $25 domestic outgoing, $45 international
- Premium Checking: FREE for all wire transfers
...

📊 Confidence: 95%
⏱️  Response time: 1.23s
📚 Sources:
  1. wire_transfer_guide.txt
```

### 2. Python API Usage

```python
from main import BankRAGSystem

# Initialize system
rag_system = BankRAGSystem()

# Ask a question
response = rag_system.query("How do I apply for a loan?")

# Access response
print(response.answer)           # The generated answer
print(response.confidence)       # Confidence score (0-1)
print(response.sources)          # List of source documents
print(response.is_refusal)       # Whether system refused to answer
print(response.response_time)    # Time taken in seconds

# Get as dictionary (JSON-serializable)
response_dict = response.to_dict()
```

### 3. Batch Processing

```python
from main import BankRAGSystem

system = BankRAGSystem()

questions = [
    "What is the interest rate on savings?",
    "How do I use mobile deposit?",
    "What are ATM fees?"
]

responses = system.batch_query(questions)

for q, r in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {r.answer[:100]}...")
    print()
```

---

## 🔌 API Integration

### FastAPI Example

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import BankRAGSystem

app = FastAPI(title="Bank RAG API")
rag_system = BankRAGSystem()

class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True

@app.post("/api/v1/query")
async def query_endpoint(request: QuestionRequest):
    """Query the RAG system"""
    try:
        response = rag_system.query(
            request.question,
            include_sources=request.include_sources
        )
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "bank-rag"}

# Run: uvicorn api:app --reload --port 8000
```

### Response Format

```json
{
  "answer": "To apply for a personal loan at TechBank...",
  "confidence": 0.92,
  "is_refusal": false,
  "response_time": 1.45,
  "sources": [
    {
      "source": "loan_policy.txt",
      "chunk_id": "chunk_42",
      "excerpt": "Personal Loan Application Process..."
    }
  ],
  "metadata": {
    "num_retrieved_docs": 5,
    "question_length": 32,
    "answer_length": 456
  }
}
```

### Frontend Integration

POST to `http://localhost:8000/api/v1/query` with JSON body:
```json
{ "question": "Your question", "include_sources": true }
```

---

## 📁 Project Structure

```
bank_RAG/
├── config.py                    # Configuration (Ollama, embeddings, RAG params)
├── document_loader.py           # Load PDF/DOCX/HTML/TXT files
├── chunking.py                  # Text splitting and preprocessing
├── embeddings.py                # sentence-transformers embeddings generation
├── retriever.py                 # Hybrid retrieval (BM25 + Vector)
├── rag_pipeline.py              # Main RAG logic + citation + refusal
├── evaluation.py                # Evaluation metrics (Hit Rate, MRR, nDCG)
├── main.py                      # Entry point and demos
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── documents/                   # Banking documents (knowledge base)
│   ├── loan_policy.txt
│   ├── credit_card_faq.txt
│   ├── account_types.txt
│   ├── atm_locations.txt
│   ├── wire_transfer_guide.txt
│   └── mobile_banking_guide.txt
│
├── data/                        # Generated data
│   └── vector_store/            # FAISS index (auto-generated)
│       ├── index.faiss
│       └── index.pkl
│
└── tests/                       # Test files and results
    └── evaluation_results.json  # Evaluation metrics output
```

---

## ⚙️ Configuration

### .env file (optional configuration)

Create a `.env` file in the project root to customize models (optional, defaults work):

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=phi3:mini  # Lightweight (3.8B, ~4GB RAM) - default
# OLLAMA_LLM_MODEL=llama3.1:8b  # Better quality (8GB RAM)

# Embeddings Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Lightweight (384 dims)
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed configuration options.

### config.py (system parameters)

```python
# RAG Parameters
chunk_size = 1000              # Characters per chunk
chunk_overlap = 200            # Overlap between chunks
top_k = 5                      # Number of chunks to retrieve
refusal_threshold = 0.3        # Minimum confidence to answer
```

### Optional: Cohere Reranking

```bash
# Add to .env file
COHERE_API_KEY=your-cohere-key
```

---

## 📄 Adding New Documents

### Supported Formats
- `.txt` - Plain text
- `.pdf` - PDF documents
- `.docx` - Microsoft Word
- `.html` - HTML files

### Steps

1. **Add document to documents/ folder**
   ```bash
   cp /path/to/new_policy.pdf documents/
   ```

2. **Rebuild vector store**
   ```python
   # Documents are automatically loaded on next run
   python main.py interactive
   ```
   
   Or programmatically:
   ```python
   from document_loader import load_banking_documents
   from chunking import chunk_banking_documents
   from embeddings import create_embeddings_generator
   from retriever import create_vector_retriever
   
   # Reload documents
   documents = load_banking_documents("./documents")
   chunks = chunk_banking_documents(documents)
   embeddings = create_embeddings_generator()
   
   # Rebuild vector store
   retriever, vector_store = create_vector_retriever(
       chunks,
       embeddings.get_embeddings_model(),
       save_path="./data/vector_store"
   )
   ```

3. **Update evaluation** (optional)
   Add test queries in `evaluation.py`:
   ```python
   {
       'query': 'New question about the new document?',
       'expected_sources': ['new_policy.pdf']
   }
   ```

---

## 📊 Evaluation

### Run Evaluation

```bash
# Quick test (2 min) - verify system works
python3 test_evaluation_quick.py

# Full evaluation (5-10 min) - comprehensive metrics
python3 evaluation.py

# View HTML report
firefox ./tests/evaluation_report.html
```

### Evaluation Metrics (9 indicators)

| Type | Metrics | Target | Why Important |
|------|---------|--------|---------------|
| **🔍 Retrieval** | Precision, Recall, Hit Rate, MRR, nDCG | ≥90% | Found right documents? |
| **✨ Generation** | Faithfulness, Answer Relevancy | ≥95% | No hallucination? Answers question? |
| **📈 Accuracy** | Answer Correctness, Similarity | ≥80% | Matches reference answer? |

**Key Metrics**:
- **Faithfulness** 🔥: Detects if AI invents information (e.g., wrong interest rates). Target: ≥95%
- **Answer Relevancy**: Checks if answer actually addresses the question. Target: ≥85%
- **Precision**: Percentage of retrieved documents that are relevant. Target: ≥90%

### Test Dataset

30 test questions covering 5 banking categories:
- 💰 Personal Loans (6) | 💳 Credit Cards (5) | 🏦 Accounts (6) 
- 🌍 Transfers (6) | 📱 Mobile/ATM (7)

Each includes: question, expected sources, reference answer, and category.

### Sample Output

```
📊 COMPREHENSIVE RAG EVALUATION RESULTS
Evaluated: 30 queries

🔍 RETRIEVAL QUALITY
  Precision:        91.67%  [✅ PASS ≥90%]
  Context Recall:   88.50%

✨ GENERATION QUALITY  
  Faithfulness:     96.20%  [✅ PASS ≥95%] ← No hallucination
  Answer Relevancy: 89.30%  [✅ PASS ≥85%] ← Answers question

📈 PRD Requirements: All Passed ✅
```

### Advanced Usage

<details>
<summary>Click to expand advanced options</summary>

**Skip LLM evaluation (faster, cheaper)**:
```python
evaluator = ComprehensiveRAGEvaluator(pipeline, use_ragas=False)
```

**Evaluate single query**:
```python
result = evaluator.evaluate_query(
    query="How do I apply for a loan?",
    expected_sources=['loan_policy.txt'],
    reference_answer="Apply online or visit branch..."
)
```

**Add custom test questions** in `evaluation.py`:
```python
{
    'query': 'Your question?',
    'expected_sources': ['doc.txt'],
    'reference_answer': 'Answer...',
    'category': 'loan'
}
```

**Cost**: ~$1-2 for 30 questions (GPT-4) or $0.2-0.5 (GPT-4o-mini)
</details>

---

## 📦 Requirements

### Python Packages (Latest Versions)

```
# Core RAG Framework
langchain >= 0.3.0
langchain-openai >= 0.2.0
langchain-community >= 0.3.0
openai >= 1.50.0

# Retrieval & Vector Store
faiss-cpu >= 1.8.0
rank-bm25 >= 0.2.2

# Evaluation Framework
ragas >= 0.1.0

# Document Processing
pypdf >= 5.0.0
python-docx >= 1.1.2

# Data & Utils
numpy >= 1.26.0
pandas >= 2.2.0
pydantic >= 2.9.0
python-dotenv >= 1.0.0
```

See `requirements.txt` for complete list.

**Note**: The project uses the latest stable versions of all libraries (as of October 2024) for best performance and security.

### System Requirements

- Python 3.8+
- **4GB+ RAM** (lightweight config) or 8GB+ RAM (standard config)
- Ollama installed and running locally
- ~5GB disk space (lightweight) or ~10GB (standard)

**💡 Performance Tips**: See [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) for low-resource optimization

---

## 🔧 Troubleshooting

### Issue: "No module named 'langchain'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Ollama connection failed"

**Solution:**
1. Ensure Ollama is installed and running
2. Check Ollama is accessible at configured URL (default: http://localhost:11434)
3. Verify model is downloaded: `ollama list`
```bash
# Start Ollama (if not running)
ollama serve

# Download model if missing
ollama pull llama3.1:8b
```

### Issue: "FAISS index not found"

**Solution:**
Vector store is auto-created on first run. Just run:
```bash
python main.py interactive
```

### Issue: "Response time > 3 seconds"

**Possible causes:**
- First query (cold start): Normal, subsequent queries are faster
- Large document corpus: Consider reducing chunk size
- Slow CPU: Consider using GPU or smaller model
- Ollama not running: Ensure Ollama service is active

**Optimization:**
```python
# config.py
chunk_size = 800  # Reduce from 1000
top_k = 3         # Reduce from 5
```

### Issue: "Low confidence scores"

**Solutions:**
1. Add more relevant documents to knowledge base
2. Improve document quality and formatting
3. Adjust confidence threshold:
```python
# config.py
refusal_threshold = 0.2  # Lower threshold (was 0.3)
```

### Issue: "Module not found" errors

**Solution:**
Ensure virtual environment is activated:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# Then reinstall
pip install -r requirements.txt
```

---

## 🧪 Testing

### Unit Tests (Optional)

```bash
# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

### Manual Testing Checklist

- [ ] System initializes without errors
- [ ] Can query simple questions
- [ ] Sources are cited correctly
- [ ] Confidence scores are reasonable
- [ ] Refusal mechanism works (try unrelated questions)
- [ ] Response time < 3 seconds
- [ ] Vector store persists between runs

---

## 📈 Performance Monitoring

### Log Analysis

Logs show timing for each step:
```
INFO - Retrieved 5 documents in 0.23s
INFO - Generated answer in 1.02s
INFO - Response generated in 1.45 seconds
```

### Metrics to Track

- **Response Time**: Target < 3 seconds
- **Confidence**: Average should be > 0.7
- **Refusal Rate**: Should be < 15%
- **Hit Rate**: Should be > 85%

---

## 📊 Production Monitoring & Optimization

Simple yet powerful monitoring system to track performance and optimize with Ragas metrics.

### Quick Start

**1. Automatic Logging** (default: enabled)

```python
system = BankRAGSystem(enable_monitoring=True)  # Default
response = system.query("What are the interest rates?")
# Query is automatically logged to logs/queries/queries_2025-11-09.jsonl
```

**2. Check Session Stats**

```bash
python main.py interactive

💬 Your question: stats

📊 Session Summary
════════════════════════════════════════════════════════════
Total Queries:       15
Avg Response Time:   1,234 ms
Avg Retrieval Score: 0.845
Refused:             2 (13.3%)
════════════════════════════════════════════════════════════
```

**3. Analyze with Ragas**

```bash
# Analyze all logs and get recommendations
python analyze_logs.py
```

### What Gets Logged

- User query and system answer
- Response time and retrieval scores
- Retrieved document contexts (for Ragas evaluation)
- Refusal decisions
- Daily log rotation: `logs/queries/queries_YYYY-MM-DD.jsonl`

### Analysis Report

The analyzer provides:

**Basic Stats:**
- Response time (avg, p95)
- Retrieval quality scores
- Slow/poor performing queries

**Ragas Metrics:**
- **Faithfulness** (0-1): Does answer stay true to documents?
- **Answer Relevancy** (0-1): Does answer address the question?

Both metrics target >0.7 for production quality.

**Optimization Recommendations:**

Prioritized suggestions based on detected issues:

| Issue | Recommendation |
|-------|----------------|
| Slow response (>3s) | Reduce k parameter, optimize index |
| Poor retrieval (<0.6) | Adjust chunk_size (800) and overlap (200) |
| Low faithfulness (<0.7) | Strengthen prompt, lower temperature |
| Low relevancy (<0.7) | Improve prompt clarity, check retrieval |

### Optimization Workflow

```bash
# Week 1: Collect data
python main.py interactive
# → Use system, ask questions

# Week 2: Analyze
python analyze_logs.py
# → See: "Poor retrieval: 15% queries <0.6 score"
# → Recommendation: Adjust chunk_size to 800

# Week 3: Implement
# Edit chunking.py: chunk_size=800, overlap=200

# Week 4: Verify
python analyze_logs.py
# → Compare: Score improved 0.65 → 0.82 ✓
```

### Configuration

```python
# Disable monitoring
system = BankRAGSystem(enable_monitoring=False)

# Custom log directory
from monitoring import create_monitor
monitor = create_monitor(log_dir="custom/logs")
```

### Files

- **monitoring.py** (~140 lines): Simple query logger
- **analyze_logs.py** (~240 lines): Analyzer with Ragas integration
- Logs excluded from Git via `.gitignore`

---

## 🔐 Security Notes

### API Key Management

⚠️ **IMPORTANT**: Never commit API keys to version control!

```bash
# Add to .gitignore
echo "config.py" >> .gitignore
echo ".env" >> .gitignore
```

Use environment variables in production:
```python
import os
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
```

---

## 🤝 For Full-Stack Team

### Integration Points

1. **REST API**: Use FastAPI example above
2. **Direct Python**: Import `BankRAGSystem` class
3. **Microservice**: Deploy as separate service

### Key Classes

- `BankRAGSystem`: Main interface (use this!)
- `RAGResponse`: Structured response object
- `BankRAGPipeline`: Core RAG logic

### Response Schema

```python
@dataclass
class RAGResponse:
    answer: str              # Generated answer
    sources: List[Dict]      # Source citations
    confidence: float        # 0-1 score
    is_refusal: bool         # Whether declined to answer
    response_time: float     # Seconds taken
    metadata: Dict           # Additional info
```

---

## 📞 Support

For questions or issues:
- **Data Engineering Team**: Check code comments and docstrings
- **Ollama**: https://ollama.com/docs
- **LangChain**: https://python.langchain.com/docs/
- **Migration Guide**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## 📝 License

Internal TechBank project. Proprietary and confidential.

---

## 🎯 Next Steps

1. ✅ Test the system with sample queries
2. ✅ Review evaluation metrics
3. 🔄 Add more banking documents as needed
4. 🔄 Integrate with your API/frontend
5. 🔄 Monitor performance in production
6. 🔄 Fine-tune parameters based on user feedback

---

---

## 📝 Project Notes

### Australian Banking Content
This RAG system is configured for **Melbourne First Bank** with Australian banking terminology:
- Currency: AUD (not USD)
- Payment systems: PayID, BPAY, Osko (not Zelle, ACH)
- Regulations: AUSTRAC, APRA (not FDIC, FINRA)
- Locations: Melbourne, Geelong, Ballarat (Victoria)
- Phone: 13 MELB (13 6352) - Australian format

### Latest Libraries
All packages use the latest stable versions (October 2024):
- LangChain 0.3.x for improved performance
- sentence-transformers 2.2+ for embeddings
- Ollama for local LLM inference
- No deprecation warnings
- Production-ready and future-proof

### Code Quality
- ✅ 0 linter errors
- ✅ Clean imports (langchain_community)
- ✅ Type hints throughout
- ✅ Comprehensive documentation
- ✅ Modular architecture

---

**Built with ❤️ by the Data Engineering Team for Melbourne Projects**
