import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paperrag.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set logging level for paperrag.parsers.figure_extractor to ERROR
# figure_extractor_logger = logging.getLogger('paperrag.parsers.figure_extractor')
# figure_extractor_logger.setLevel(logging.DEBUG)