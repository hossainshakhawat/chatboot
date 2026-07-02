# Worker Module Setup Guide

## Overview

The `worker` module provides functionality for:
1. Reading PDF documents
2. Extracting and chunking text
3. Generating BERT embeddings using Spacy and sentence-transformers
4. Storing embeddings in Weaviate database
5. Searching for similar documents

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Spacy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Start Weaviate (if not already running)

```bash
docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
```

### 4. Use Django Management Command

Process a PDF file:
```bash
python manage.py process_pdfs --file /path/to/document.pdf
```

Process all PDFs in a directory:
```bash
python manage.py process_pdfs --directory /path/to/pdfs/
```

Search:
```bash
python manage.py process_pdfs --search "search query"
```

## File Structure

```
worker/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration settings
├── pdf_reader.py              # PDF extraction and chunking
├── embedder.py                # BERT embeddings and Spacy NLP
├── weaviate_client.py         # Weaviate database client
├── worker.py                  # Main orchestrator
├── example_usage.py           # Usage examples
├── tests.py                   # Unit tests
└── README.md                  # Detailed documentation

chatbot/management/commands/
└── process_pdfs.py            # Django management command
```

## Key Components

### PDFReader (`pdf_reader.py`)
- Extracts text from PDF files using pdfplumber
- Chunks text with configurable overlap
- Returns text with page numbers

### Embedder (`embedder.py`)
- Uses sentence-transformers for BERT embeddings
- Integrates Spacy for NLP processing
- Supports batch embedding generation

### WeaviateClient (`weaviate_client.py`)
- Manages connection to Weaviate database
- Stores document chunks with embeddings
- Performs vector similarity search
- Supports batch operations

### PDFWorker (`worker.py`)
- Orchestrates the entire pipeline
- Processes single PDFs or directories
- Handles search queries
- Manages database operations

## Configuration

Edit `worker/config.py` to customize:

```python
# Weaviate settings
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight BERT model

# Text chunking
CHUNK_SIZE = 512               # Characters per chunk
CHUNK_OVERLAP = 50             # Overlap between chunks

# Spacy model
SPACY_MODEL = "en_core_web_sm"

# Weaviate class name
CLASS_NAME = "Document"
```

## Usage Examples

### Python Code

```python
from worker.worker import PDFWorker

# Initialize worker
worker = PDFWorker()

# Process single PDF
uuids = worker.process_pdf("document.pdf")
print(f"Stored {len(uuids)} chunks")

# Process directory
stats = worker.process_directory("pdfs/")
print(f"Processed {stats['successful']} files")

# Search
results = worker.search("machine learning", limit=5)
for result in results:
    print(f"Source: {result['source']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Distance: {result['distance']:.4f}\n")

# Clean up
worker.close()
```

### Django Management Command

```bash
# Process file
python manage.py process_pdfs --file document.pdf

# Process directory
python manage.py process_pdfs --directory pdfs/

# Search
python manage.py process_pdfs --search "your query"

# Clear database
python manage.py process_pdfs --clear
```

## Database Schema

The Weaviate `Document` class has the following properties:

- **content** (TEXT): The document chunk text
- **source** (TEXT): Path to the source PDF file
- **page** (INT): Page number in the PDF
- **chunk_index** (INT): Index of the chunk in the document
- **vector**: BERT embedding (automatically managed by Weaviate)

## Troubleshooting

### Weaviate Connection Issues
```
Error: Failed to connect to Weaviate
```
- Ensure Weaviate is running: `docker ps | grep weaviate`
- Check host and port in `config.py`
- Verify Weaviate is accessible: `curl http://localhost:8080/v1/meta`

### Spacy Model Missing
```
Error: Can't find model "en_core_web_sm"
```
- Download: `python -m spacy download en_core_web_sm`

### Memory Issues
- Reduce `CHUNK_SIZE` in `config.py`
- Process files individually instead of batch
- Use smaller embedding model if needed

### No Results in Search
- Ensure PDFs were processed correctly
- Check Weaviate contains documents: query GraphQL at http://localhost:8080/v1/graphql
- Verify embeddings were generated

## Performance Tips

1. **Batch Processing**: Use `process_directory()` for multiple files
2. **Chunk Size**: Smaller chunks = more vectors but faster search
3. **Embedding Model**: "all-MiniLM-L6-v2" is lightweight but accurate
4. **Spacy NLP**: Optional, only needed for entity extraction

## GraphQL Interface

Access Weaviate GraphQL at: http://localhost:8080/v1/graphql

Example query to count documents:
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

## Running Tests

```bash
python -m unittest worker.tests
```

## Next Steps

1. Prepare your PDF files
2. Run: `python manage.py process_pdfs --directory /path/to/pdfs/`
3. Search: `python manage.py process_pdfs --search "your query"`
4. Integrate search results into your Django views

## Support

For issues or questions, check:
- `worker/README.md` - Detailed API documentation
- `worker/example_usage.py` - Complete usage examples
- Docker logs: `docker logs weaviate`
