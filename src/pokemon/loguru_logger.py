from loguru import logger

logger.remove()
logger.add("pokemon.log", rotation="10 MB")
