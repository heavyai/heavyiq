from io import BytesIO, FileIO
from typing import Optional

from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader


class CustomPDFReader(PDFReader):
    """
    Overrided PDFReader class which adds support for reading pdf content from database blobs type.
    """

    def load_data_from_file_object(
        self,
        file_obj: BytesIO | FileIO,
        file_name: str,
        extra_info: dict | None = None,
    ) -> list[Document]:
        """Parse file object."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required to read PDF files: `pip install pypdf`")
        pdf = pypdf.PdfReader(file_obj)

        # Get the number of pages in the PDF document
        num_pages = len(pdf.pages)

        docs = []

        # This block returns a whole PDF as a single Document
        if self.return_full_document:
            text = ""
            metadata = {"file_name": file_name}

            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                text += page_text

            if extra_info is not None:
                metadata.update(extra_info)

            docs.append(Document(text=text, metadata=metadata))  # type: ignore

        # This block returns each page of a PDF as its own Document
        else:
            # Iterate over every page

            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]

                metadata = {"page_label": page_label, "file_name": file_name}
                if extra_info is not None:
                    metadata.update(extra_info)

                docs.append(Document(text=page_text, metadata=metadata))  # type: ignore

        return docs
