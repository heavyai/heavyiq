import os

import promptlayer

from heavynl.config import get_config

is_promptlayer_active = False

config = get_config()
if config.promptlayer_api_key is not None and config.promptlayer_api_key != "":
    os.environ["PROMPTLAYER_API_KEY"] = config.promptlayer_api_key
    promptlayer.api_key = config.promptlayer_api_key
    is_promptlayer_active = True
