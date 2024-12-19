from figure_extractor.figure_extractor import extract_figures
import requests
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
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

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing figures and their metadata for each document.
        """
        try:
            figure_response = extract_figures(self.pdf_path, self.output_dir)
            for doc_i, figure_doc in enumerate(figure_response):
                for fig_i, figure in enumerate(figure_doc["figures_with_metadata"]):
                    image_text = " ".join(figure["metadata"]["imageText"])
                    figure_response[doc_i]["figures_with_metadata"][fig_i]["metadata"]["imageText"] = image_text
            logger.info("Figures extracted successfully.")
            return figure_response
        except Exception as e:
            logger.error(f"Error extracting figures: {e}")
            raise