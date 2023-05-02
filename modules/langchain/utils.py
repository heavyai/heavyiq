import os

from modules.config import config

is_promptlayer_active = False

if config.promptlayer_api_key is not None and config.promptlayer_api_key != "":
    os.environ["PROMPTLAYER_API_KEY"] = config.promptlayer_api_key
    is_promptlayer_active = True
