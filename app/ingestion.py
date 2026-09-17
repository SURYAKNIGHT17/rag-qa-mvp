from typing import List, Tuple
import pymupdf


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from PDF bytes using PyMuPDF.
    Returns a list of tuples: (page_number_1_based, page_text).
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Corrupt or invalid PDF file: {str(e)}")

    if doc.page_count == 0:
        raise ValueError("PDF file contains no pages.")

    pages_text: List[Tuple[int, str]] = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text").strip()
        if text:
            pages_text.append((page_idx + 1, text))

    doc.close()

    if not pages_text:
        raise ValueError("PDF file contains no extractable text.")

    return pages_text


def extract_text_from_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from TXT bytes.
    Returns a list with a single tuple: (1, full_text).
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception as e:
            raise ValueError(f"Failed to decode text file: {str(e)}")

    text = text.strip()
    if not text:
        raise ValueError("Text file is empty or contains no readable text.")

    return [(1, text)]


def extract_document(filename: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Routes document extraction based on file extension.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    lower_filename = filename.lower()
    if lower_filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_filename.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension for '{filename}'. Only .pdf and .txt are supported.")
