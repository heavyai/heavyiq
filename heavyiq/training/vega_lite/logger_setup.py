import logging

START_EMOJI = "🚀"
SUCCESS_EMOJI = "✅"
FAILURE_EMOJI = "❌"
logger = logging.getLogger("langgraph")
logger.setLevel(logging.INFO)

# Optional: format + console output
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console)
