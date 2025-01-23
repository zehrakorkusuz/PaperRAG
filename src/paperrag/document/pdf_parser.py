import asyncio
from llmsherpa.readers import LayoutPDFReader
import os
import logging

logger = logging.getLogger(__name__)

def is_valid_document_path(document_path):
    if os.path.isdir(document_path):
        raise ValueError(f"The path {document_path} is a directory, not a file.")
    if not os.path.isfile(document_path):
        raise FileNotFoundError(f"The file {document_path} does not exist.")
    if not document_path.endswith('.pdf'):
        raise ValueError(f"The file {document_path} is not a PDF.")

def call_llmsherpa(document_path):
    llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(document_path)
    ## update metadata
    doc.source = os.path.basename(document_path)
    return doc

async def process_pdf(pdf_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_llmsherpa, pdf_path)

async def parse_files(path):
    """Process PDF files and return list of results"""
    if not os.path.exists(path):
        logging.error(f"Path does not exist: {path}")
        return []

    try:
        if os.path.isfile(path):
            if not path.endswith('.pdf'):
                logging.warning(f"Not a PDF file: {path}")
                return []
            result = await process_pdf(path)
            return [result] if result else []

        # Handle directory
        pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
        if not pdf_files:
            logging.warning(f"No PDF files found in directory: {path}")
            return []

        results = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(path, pdf_file)
            result = await process_pdf(pdf_path)
            if result:
                results.append(result)
                
        return results

    except Exception as e:
        logging.error(f"Error parsing files: {e}")
        return []