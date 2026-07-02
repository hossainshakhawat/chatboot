"""Django management command for processing PDFs with the worker module."""

from django.core.management.base import BaseCommand, CommandError
import os
from worker.worker import PDFWorker


class Command(BaseCommand):
    help = "Process PDF files and store embeddings in Weaviate"

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", type=str, help="Path to a single PDF file to process"
        )
        parser.add_argument(
            "--directory",
            type=str,
            help="Path to a directory containing PDF files to process",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear all documents from Weaviate before processing",
        )
        parser.add_argument(
            "--search",
            type=str,
            help="Search for documents similar to the given query",
        )

    def handle(self, *args, **options):
        try:
            worker = PDFWorker()

            if options["clear"]:
                self.stdout.write("Clearing Weaviate database...")
                worker.clear_database()
                self.stdout.write(self.style.SUCCESS("Database cleared"))

            if options["file"]:
                file_path = options["file"]
                if not os.path.exists(file_path):
                    raise CommandError(f"File not found: {file_path}")

                self.stdout.write(f"Processing file: {file_path}")
                uuids = worker.process_pdf(file_path)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully processed file. Stored {len(uuids)} chunks"
                    )
                )

            elif options["directory"]:
                directory_path = options["directory"]
                if not os.path.isdir(directory_path):
                    raise CommandError(f"Directory not found: {directory_path}")

                self.stdout.write(f"Processing directory: {directory_path}")
                stats = worker.process_directory(directory_path)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Processed {stats['successful']} files, "
                        f"{stats['failed']} failed"
                    )
                )

            elif options["search"]:
                query = options["search"]
                self.stdout.write(f"Searching for: {query}")
                results = worker.search(query, limit=10)

                if not results:
                    self.stdout.write("No results found")
                else:
                    for i, result in enumerate(results, 1):
                        self.stdout.write(f"\n--- Result {i} ---")
                        self.stdout.write(f"Source: {result['source']}")
                        self.stdout.write(f"Page: {result['page']}")
                        self.stdout.write(
                            f"Content: {result['content'][:100]}..."
                        )
                        self.stdout.write(f"Distance: {result['distance']:.4f}")

            worker.close()

        except Exception as e:
            raise CommandError(f"Error: {str(e)}")
