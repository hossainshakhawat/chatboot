# Worker Module - Quick Start Guide

## 1. Install & Setup (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Download Spacy model
python -m spacy download en_core_web_sm

# Start Weaviate (in another terminal)
docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
```

## 2. Process PDFs

### Single File
```bash
python manage.py process_pdfs --file path/to/document.pdf
```

### Multiple Files (Directory)
```bash
python manage.py process_pdfs --directory path/to/pdf/folder/
```

## 3. Search Documents

```bash
python manage.py process_pdfs --search "your search query here"
```

## 4. Python Code Usage

```python
from worker.worker import PDFWorker

# Setup
worker = PDFWorker()

# Process PDF
worker.process_pdf("document.pdf")

# Search
results = worker.search("your query", limit=5)
for r in results:
    print(f"{r['source']} (page {r['page']}): {r['content'][:100]}...")

# Cleanup
worker.close()
```

## 5. Django View Integration

```python
# views.py
from worker.worker import PDFWorker
from django.shortcuts import render

def search_documents(request):
    query = request.GET.get('q', '')
    results = []
    
    if query:
        worker = PDFWorker()
        results = worker.search(query, limit=10)
        worker.close()
    
    return render(request, 'search.html', {'results': results, 'query': query})
```

## 6. Configuration

Edit `worker/config.py`:

```python
# Change Weaviate location if needed
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080

# Adjust text chunking
CHUNK_SIZE = 512        # Increase for longer context
CHUNK_OVERLAP = 50      # Increase for more overlap

# Change embedding model (if you want)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # More accurate
```

## 7. Management Commands Reference

| Command | Purpose |
|---------|---------|
| `process_pdfs --file PDF` | Process single PDF |
| `process_pdfs --directory DIR` | Process all PDFs in directory |
| `process_pdfs --search QUERY` | Search processed documents |
| `process_pdfs --clear` | Delete all documents from Weaviate |

## 8. File Locations

- **Worker Module**: `/worker/`
- **Main Class**: `/worker/worker.py`
- **Django Command**: `/chatbot/management/commands/process_pdfs.py`
- **Setup Guide**: `WORKER_SETUP.md`
- **Full Docs**: `worker/README.md`

## 9. Check Weaviate

Access GraphQL console: http://localhost:8080/v1/graphql

Query to check documents:
```graphql
{
  Aggregate {
    Document {
      meta {
        count
      }
    }
  }
}
```

## 10. Troubleshooting

**Problem**: "Connection refused"
```
Solution: docker ps | grep weaviate
(Make sure Weaviate is running)
```

**Problem**: "Can't find model"
```
Solution: python -m spacy download en_core_web_sm
```

**Problem**: Out of memory
```
Solution: Reduce CHUNK_SIZE in worker/config.py
```

## Example Workflow

```bash
# 1. Create a folder with PDF files
mkdir my_pdfs
cp document1.pdf document2.pdf my_pdfs/

# 2. Process all PDFs
python manage.py process_pdfs --directory my_pdfs/

# 3. Search
python manage.py process_pdfs --search "important topic"

# Output:
# --- Result 1 ---
# Source: my_pdfs/document1.pdf
# Page: 3
# Content: ... relevant text ...
# Distance: 0.2341
```

## API Quick Reference

### PDFWorker Methods
```python
worker.process_pdf(filepath)           # Process single PDF
worker.process_directory(dirpath)      # Process directory
worker.search(query, limit=10)         # Search documents
worker.clear_database()                # Clear all data
worker.close()                         # Cleanup
```

### Return Format
```python
results = worker.search("query")
# Returns:
# [{
#     'id': 'uuid',
#     'content': 'text chunk...',
#     'source': 'path/to/pdf.pdf',
#     'page': 3,
#     'chunk_index': 5,
#     'distance': 0.234
# }, ...]
```

## Common Use Cases

### Use Case 1: Build a Document QA System
```python
def ask_documents(question):
    worker = PDFWorker()
    context = worker.search(question, limit=3)
    # Use context with LLM to answer
    worker.close()
    return answer
```

### Use Case 2: Document Recommender
```python
def recommend_documents(user_interest):
    worker = PDFWorker()
    docs = worker.search(user_interest, limit=5)
    return [d['source'] for d in docs]
```

### Use Case 3: Full-Text Search Replacement
```python
def search_view(request):
    query = request.GET.get('q')
    worker = PDFWorker()
    results = worker.search(query, limit=20)
    worker.close()
    return JsonResponse(results, safe=False)
```

## Performance Tips

1. **For Large Batches**: Process directories instead of individual files
2. **For Fast Search**: Use default "all-MiniLM-L6-v2" model
3. **For Accuracy**: Increase CHUNK_OVERLAP
4. **For Memory**: Reduce CHUNK_SIZE

## Next: Detailed Guides

- Setup Details: See `WORKER_SETUP.md`
- API Reference: See `worker/README.md`
- Implementation: See `WORKER_MODULE_SUMMARY.md`

---

**Ready to get started? Process your first PDF in 30 seconds!**

```bash
python manage.py process_pdfs --file your_document.pdf
python manage.py process_pdfs --search "topic"
```
