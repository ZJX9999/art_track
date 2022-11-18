import logging.config
import logging.handlers

import yaml

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
sh = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[sh])
log = logging.getLogger('ArtTrack')


def setup_log(file=None):
    with open(file=file or 'log.yaml', mode='r', encoding="utf-8") as f:
        logging_yaml = yaml.load(stream=f, Loader=yaml.FullLoader)
        logging.config.dictConfig(config=logging_yaml)
