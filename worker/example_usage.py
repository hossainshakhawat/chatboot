"""Example usage of the PDF Worker module."""

from worker import PDFWorker


def main():
    """Example usage of PDFWorker."""

    # Initialize the worker
    worker = PDFWorker()

    # Example 1: Process a single PDF
    pdf_path = "path/to/your/document.pdf"
    try:
        uuids = worker.process_pdf(pdf_path)
        print(f"Processed PDF, stored {len(uuids)} chunks")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    # Example 2: Process all PDFs in a directory
    directory_path = "path/to/pdf/directory"
    try:
        stats = worker.process_directory(directory_path)
        print(f"Processed {stats['successful']} files, {stats['failed']} failed")
    except NotADirectoryError as e:
        print(f"Error: {e}")

    # Example 3: Search for similar documents
    query = "Your search query here"
    results = worker.search(query, limit=5)

    for result in results:
        print(f"\n--- Result ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Content: {result['content'][:200]}...")
        print(f"Distance: {result['distance']}")

    # Clean up
    worker.close()


if __name__ == "__main__":
    main()
