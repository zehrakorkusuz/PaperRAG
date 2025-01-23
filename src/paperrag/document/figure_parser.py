from paperrag.parsers.figure_extractor.figure_extractor import extract_figures
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FigureExtractor:
    """
    FigureExtractor is a class to extract figures from PDF files.

    Attributes:
        pdf_path (str): Path to the PDF file or a folder containing PDF files.
        output_dir (str): Directory where the extracted figures will be saved.
        url (str): URL of the figure extraction service.

    Methods:
        __init__(pdf_path: str, output_dir: str):
            Initializes the FigureExtractor with the given PDF path and output directory.
        check_service() -> bool:
            Checks if the figure extraction service is running.
        extract() -> List[Dict[str, Any]]:
            Extracts figures from the PDF file(s) and returns the response with metadata.
    """
    SERVICE_URL = "http://localhost:5001/api/docs"
    SUCCESS_STATUS_CODE = 200

    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.url = self.SERVICE_URL

    def check_service(self) -> bool:
        """
        Checks if the figure extraction service is running.

        Returns:
            bool: True if the service is running, False otherwise.
        """
        try:
            response = requests.get(self.url)
            if response.status_code == self.SUCCESS_STATUS_CODE:
                logger.info("Figure extraction service is running.")
                return True
            else:
                logger.error(f"Service returned status code {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking service: {e}")
            return False

    def extract(self) -> List[Dict[str, Any]]:
        """
        Extracts figures from the PDF file(s) and returns the response with metadata.
        """
        try:
            figure_response = extract_figures(self.pdf_path, self.output_dir)
            
            # Verify figure_response is a list
            if not isinstance(figure_response, list):
                logger.error(f"Expected figure_response to be a list, got {type(figure_response)}")
                raise TypeError("figure_response should be a list of dictionaries.")
            
            for doc_i, figure_doc in enumerate(figure_response):
                logger.debug(f"Processing document {doc_i}: {figure_doc}")
                
                # Verify figure_doc is a dictionary
                if not isinstance(figure_doc, dict):
                    logger.error(f"Expected figure_doc to be a dict, got {type(figure_doc)}")
                    raise TypeError("Each figure_doc should be a dictionary.")
                
                if "figures_with_metadata" in figure_doc:
                    figures_with_metadata = figure_doc.get("figures_with_metadata", [])
                    
                    # Verify figures_with_metadata is a list
                    if not isinstance(figures_with_metadata, list):
                        logger.error(f"'figures_with_metadata' is not a list in document {doc_i}: {type(figures_with_metadata)}")
                        raise TypeError("'figures_with_metadata' should be a list of dictionaries.")
                    
                    for fig_i, figure in enumerate(figures_with_metadata):
                        if not isinstance(figure, dict):
                            logger.error(f"Expected figure to be a dict, got {type(figure)} in document {doc_i}, figure {fig_i}")
                            raise TypeError("Each figure entry should be a dictionary.")
                        
                        if "metadata" in figure and "imageText" in figure["metadata"]:
                            image_text = " ".join(figure["metadata"]["imageText"])
                            figure_response[doc_i]["figures_with_metadata"][fig_i]["metadata"]["imageText"] = image_text
            logger.info("Figures extracted successfully.")
            return figure_response
        except Exception as e:
            logger.error(f"Error extracting figures: {e}")
            raise