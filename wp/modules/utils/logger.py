import os, sys, logging

def setup_logger(debug, name="Default", file=None, file_level=logging.DEBUG, stream_level=logging.CRITICAL, path=None):
    """
        Setup a logger for the active learning loop

        Parameters:
            propagate (bool): activate logging output in console?
    """

    logger = logging.Logger(name)

    # Stream handler
    log_level = logging.DEBUG if debug else stream_level
    logger.handler = logging.StreamHandler(sys.stdout)
    logger.handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handler.setFormatter(formatter)
    logger.addHandler(logger.handler)

    # Add file handler?
    if not (file is None):

        log_path = None
        if path is None:
            dir_name = os.path.dirname(os.path.realpath(__file__))
            log_path = os.path.join(dir_name, "logs", file) 

        else:
            log_path = os.path.join(path, file)    

        fh = logging.FileHandler(log_path)
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        
    return logger