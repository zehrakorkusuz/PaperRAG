import asyncio
from llmsherpa.readers import LayoutPDFReader
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

async def process_all_pdfs_in_folder(folder_path):
    """
    Processes all PDF files in the specified folder asynchronously.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list: A list of results from processing each PDF file.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        Exception: If an error occurs during the processing of any PDF file.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    results = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        logging.info(f"Processing PDF: {pdf_path}")
        result = await process_pdf(pdf_path)
        results.append(result)
        logging.info(f"Finished processing PDF: {pdf_path}")
    
    return results