import asyncio
import os
import logging
from typing import Tuple, List, Dict, Any
from pdf_parser import process_all_pdfs_in_folder
from figure_parser import FigureExtractor

class CustomLoader:
    def __init__(self, pdf_path: str = None, output_dir: str = None, async_mode: bool = True):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.async_mode = async_mode
        self.figure_extractor = FigureExtractor(pdf_path, output_dir)

    async def load_async(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load and process PDFs and extract figures with metadata asynchronously.

        Returns:
            Tuple containing the processed documents and figure responses.
        """
        doc_task = process_all_pdfs_in_folder(self.pdf_path)
        figure_task = asyncio.to_thread(self.figure_extractor.extract)

        doc, figure_response = await asyncio.gather(doc_task, figure_task)

        return doc, figure_response

    def load_sync(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load and process PDFs and extract figures with metadata synchronously.

        Returns:
            Tuple containing the processed documents and figure responses.
        """
        doc = process_all_pdfs_in_folder(self.pdf_path)
        figure_response = self.figure_extractor.extract()

        return doc, figure_response

    def load(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load and process PDFs and extract figures with metadata.

        Returns:
            Tuple containing the processed documents and figure responses.
        """
        if self.async_mode:
            try:
                return asyncio.run(self.load_async())
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self.load_async())
                else:
                    raise
        else:
            return self.load_sync()