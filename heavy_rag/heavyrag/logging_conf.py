import logging

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create logger object to be imported by other modules
logger = logging.getLogger(__name__)
