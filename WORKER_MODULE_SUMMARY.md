# Worker Module - Implementation Summary

## What Was Created

A complete `worker` module for PDF document processing with BERT embeddings and Weaviate storage.

### Module Files

**Core Module** (`/worker/`):
1. **config.py** - Configuration settings
   - Weaviate connection details
   - Embedding model selection
   - Text chunking parameters
   - Spacy model selection

2. **pdf_reader.py** - PDF text extraction
   - Extract text from PDFs using pdfplumber
   - Intelligent text chunking with overlap
   - Page tracking

3. **embedder.py** - BERT embeddings & NLP
   - BERT embeddings via sentence-transformers
   - Spacy NLP integration for entity extraction
   - Batch embedding support
   - Uses lightweight "all-MiniLM-L6-v2" model

4. **weaviate_client.py** - Weaviate database client
   - Connection management
   - Document class creation
   - Single and batch document storage
   - Vector similarity search
   - Database cleanup

5. **worker.py** - Main orchestrator
   - Coordinates all components
   - Process single PDFs or directories
   - Search functionality
   - Connection management

6. **example_usage.py** - Usage examples
   - Single file processing
   - Directory processing
   - Search examples

### Django Integration

**Management Command** (`/chatbot/management/commands/process_pdfs.py`):
```bash
python manage.py process_pdfs --file document.pdf
python manage.py process_pdfs --directory /path/to/pdfs/
python manage.py process_pdfs --search "query"
python manage.py process_pdfs --clear
```

### Documentation

1. **WORKER_SETUP.md** - Step-by-step setup guide
2. **worker/README.md** - Detailed API reference
3. **worker/tests.py** - Unit tests

## Key Features

### PDF Processing
- Extract text from PDF files
- Track page numbers
- Intelligent text chunking (configurable size & overlap)

### BERT Embeddings
- Uses sentence-transformers library
- Lightweight model: "all-MiniLM-L6-v2" (33M parameters)
- Fast inference, suitable for large documents

### Spacy NLP
- Named entity recognition
- Token analysis
- Optional preprocessing

### Weaviate Integration
- Store document chunks with embeddings
- Vector similarity search
- Batch operations for efficiency
- Automatic UUID generation

### Batch Operations
- Process multiple files efficiently
- Batch storage in Weaviate

## Dependencies Added

```
pdfplumber==0.10.3              # PDF text extraction
sentence-transformers==2.2.2    # BERT embeddings
weaviate-client==4.1.1          # Weaviate database client
spacy==3.7.2                    # NLP processing
torch==2.1.0                    # Deep learning framework
```

## System Requirements

### Running Weaviate
```bash
docker run -d -p 8080:8080 -p 50051:50051 \
  semitechnologies/weaviate:latest
```

### Spacy Model
```bash
python -m spacy download en_core_web_sm
```

## Workflow

```
PDF File
   ↓
[PDFReader] - Extract text, chunk text
   ↓
Text Chunks
   ↓
[Embedder] - Generate BERT embeddings
   ↓
(Text, Embedding) pairs
   ↓
[WeaviateClient] - Store in database
   ↓
Weaviate Database
   ↓
[Search] - Find similar documents
```

## Database Schema

**Document Class Properties:**
- `content` (TEXT) - Document chunk
- `source` (TEXT) - PDF file path
- `page` (INT) - Page number
- `chunk_index` (INT) - Chunk position
- `vector` (VECTOR) - BERT embedding

## Usage Examples

### Process a Single PDF
```python
from worker.worker import PDFWorker

worker = PDFWorker()
uuids = worker.process_pdf("document.pdf")
worker.close()
```

### Process Directory
```python
stats = worker.process_directory("pdfs/")
print(f"Processed {stats['successful']} files")
```

### Search
```python
results = worker.search("machine learning", limit=5)
for result in results:
    print(result['content'])
    print(f"Source: {result['source']}, Page: {result['page']}")
```

### Django Management Command
```bash
# Process file
python manage.py process_pdfs --file data.pdf

# Process directory
python manage.py process_pdfs --directory pdfs/

# Search
python manage.py process_pdfs --search "AI and machine learning"

# Clear database
python manage.py process_pdfs --clear
```

## Configuration Options

Edit `worker/config.py`:

```python
# Connection
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 512          # Characters
CHUNK_OVERLAP = 50        # Characters

# NLP
SPACY_MODEL = "en_core_web_sm"
```

## Next Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Start Weaviate**
   ```bash
   docker run -d -p 8080:8080 -p 50051:50051 \
     semitechnologies/weaviate:latest
   ```

3. **Process PDFs**
   ```bash
   python manage.py process_pdfs --directory /path/to/pdfs/
   ```

4. **Search Documents**
   ```bash
   python manage.py process_pdfs --search "your query"
   ```

5. **Integrate into Views** (Optional)
   ```python
   from worker.worker import PDFWorker
   
   def search_view(request):
       query = request.GET.get('q')
       worker = PDFWorker()
       results = worker.search(query)
       worker.close()
       return render(request, 'results.html', {'results': results})
   ```

## File Structure

```
chatboot/
├── worker/                          # New worker module
│   ├── __init__.py
│   ├── config.py                   # Configuration
│   ├── pdf_reader.py               # PDF extraction
│   ├── embedder.py                 # BERT embeddings
│   ├── weaviate_client.py          # Weaviate client
│   ├── worker.py                   # Main orchestrator
│   ├── example_usage.py            # Examples
│   ├── tests.py                    # Unit tests
│   └── README.md                   # API documentation
├── chatbot/
│   └── management/
│       └── commands/
│           └── process_pdfs.py     # Django command
├── requirements.txt                # Updated with dependencies
├── WORKER_SETUP.md                # Setup guide
├── WORKER_MODULE_SUMMARY.md       # This file
└── ... (other Django files)
```

## Testing

Run tests:
```bash
python -m unittest worker.tests
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Weaviate connection error | Check Docker container, verify localhost:8080 is accessible |
| Spacy model not found | Run `python -m spacy download en_core_web_sm` |
| Memory issues | Reduce CHUNK_SIZE in config.py |
| No search results | Verify PDFs were processed, check Weaviate GraphQL console |

## Performance Characteristics

- **Embedding Generation**: ~100-200ms per chunk (CPU) / ~10-50ms (GPU)
- **Text Chunking**: Negligible
- **Weaviate Search**: ~10-50ms (depends on database size)
- **Memory**: ~2-3GB for model loading

## Security Considerations

- Weaviate connection is unencrypted (localhost only)
- No authentication configured (update for production)
- PDF paths stored as-is in database

## Future Enhancements

1. Add authentication to Weaviate
2. Support for more file formats (DOCX, TXT)
3. Async processing with Celery
4. Web UI for document management
5. Multi-language support
6. Custom embedding models
7. Hybrid search (BM25 + vector)

---

**Status**: Complete and ready to use
**Last Updated**: 2024
