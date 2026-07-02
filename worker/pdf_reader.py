"""PDF reading and text extraction module."""

import pdfplumber
from typing import List, Tuple


class PDFReader:
    """Reads and extracts text from PDF documents."""

    @staticmethod
    def read_pdf(file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of tuples containing (text, page_number)
        """
        text_pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        text_pages.append((text, page_num))
        except Exception as e:
            raise IOError(f"Error reading PDF file {file_path}: {str(e)}")

        return text_pages

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks
