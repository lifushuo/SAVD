import logging

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(message)s",
                        datefmt="%H:%M:%S",
                        force=True)
