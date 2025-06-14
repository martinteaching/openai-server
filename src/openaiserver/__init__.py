import os, logging
from openaiserver.openaiserver import OpenAIServer


def run() -> None:
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
    OpenAIServer().run()
