# PDF Worker Module

A Django module for processing PDF documents, generating BERT embeddings, and storing them in Weaviate vector database.

## Features

- **PDF Text Extraction**: Extract text from PDF documents using pdfplumber
- **Text Chunking**: Intelligent chunking of text with configurable overlap
- **BERT Embeddings**: Generate embeddings using sentence-transformers
- **Spacy NLP**: Extract entities and process text with Spacy
- **Weaviate Storage**: Store and search embeddings in Weaviate vector database
- **Batch Processing**: Process multiple PDF files efficiently

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Spacy model:
```bash
python -m spacy download en_core_web_sm
```

3. Ensure Weaviate is running:
```bash
docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
```

## Configuration

Edit `worker/config.py` to customize:
- Weaviate host/port
- Embedding model (default: "all-MiniLM-L6-v2")
- Text chunk size (default: 512 characters)
- Chunk overlap (default: 50 characters)

## Usage

### Django Management Command

Process a single PDF:
```bash
python manage.py process_pdfs --file path/to/document.pdf
```

Process all PDFs in a directory:
```bash
python manage.py process_pdfs --directory path/to/pdf/directory
```

Search for similar documents:
```bash
python manage.py process_pdfs --search "your search query"
```

Clear all documents:
```bash
python manage.py process_pdfs --clear
```

### Python Code

```python
from worker.worker import PDFWorker

# Initialize
worker = PDFWorker()

# Process a single PDF
uuids = worker.process_pdf("document.pdf")

# Process directory
stats = worker.process_directory("pdf_directory/")

# Search
results = worker.search("query", limit=10)
for result in results:
    print(result["content"])
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")

# Clean up
worker.close()
```

## Module Structure

- **pdf_reader.py**: PDF extraction and text chunking
- **embedder.py**: BERT embeddings and Spacy NLP processing
- **weaviate_client.py**: Weaviate database operations
- **worker.py**: Main orchestrator combining all components
- **config.py**: Configuration settings
- **example_usage.py**: Example usage code

## API Reference

### PDFWorker

Main class for orchestrating PDF processing.

#### Methods:
- `process_pdf(file_path)`: Process a single PDF file
- `process_directory(directory_path)`: Process all PDFs in a directory
- `search(query, limit=10)`: Search for similar documents
- `clear_database()`: Clear all documents from Weaviate
- `close()`: Close all connections

### PDFReader

Handles PDF text extraction.

#### Methods:
- `read_pdf(file_path)`: Extract text and page numbers from PDF
- `chunk_text(text, chunk_size=512, overlap=50)`: Split text into chunks

### Embedder

Generates embeddings using BERT and Spacy.

#### Methods:
- `embed_text(text)`: Generate embedding for single text
- `embed_texts(texts)`: Generate embeddings for multiple texts
- `process_with_spacy(text)`: Process text with Spacy NLP

### WeaviateClient

Manages Weaviate database operations.

#### Methods:
- `create_class()`: Create Document class in Weaviate
- `store_document()`: Store a single document chunk
- `store_documents_batch()`: Store multiple documents in batch
- `search()`: Search for similar documents
- `delete_all()`: Delete all documents
- `close()`: Close connection

## Environment Variables

- `WEAVIATE_HOST`: Weaviate server host (default: localhost)
- `WEAVIATE_PORT`: Weaviate server port (default: 8080)

## Performance Notes

- Embedding generation takes time for large documents
- Use batch processing for multiple files
- Adjust CHUNK_SIZE based on your use case
- Consider increasing chunk overlap for better context

## Troubleshooting

**Weaviate Connection Error**:
- Ensure Weaviate is running on localhost:8080
- Check environment variables WEAVIATE_HOST and WEAVIATE_PORT

**Spacy Model Not Found**:
- Run: `python -m spacy download en_core_web_sm`

**Out of Memory**:
- Reduce CHUNK_SIZE in config.py
- Process files individually instead of in batch

## License

MIT
