import os, argparse, shutil
from utils import setup_logger

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# MODULES_PATH = os.path.join(DIR_PATH)


def remove_from(base_path, recursive, logger):
    """
        Remove content from given path.

        Parameters:
            base_path (str): The base path were to remove contents
            recursive (bool): Recursivly remove content?
            logger (Logger): The logger to use.
    """

    log_dir_content = os.listdir(LOGS_PATH)

    # Nothing to delete
    if len(log_dir_content) == 0:
        logger.info("Logs directory \"{}\" is empty.".format(base_path))

    for element in log_dir_content:
        full_path = os.path.join(LOGS_PATH, element)
        
        # Sub-directory?
        if os.path.isdir(full_path):
            if not args.recursive:
                logger.info("IGNORE: {}".format(full_path))

            else:
                # Recurse remove
                shutil.rmtree(full_path)
                logger.info("DELETE: {}".format(full_path))

        # File?
        else:
            os.remove(full_path)
            logger.info("DELETE: {}".format(full_path))


if __name__ == "__main__":
    """
        Clear all accumulated metrics.
    """

    # Parse received parameters
    parser = argparse.ArgumentParser(description="Clear metrics")
    parser.add_argument("-v", "--verbose", default=True, action="store_true", help="Create outputs in terminal")
    parser.add_argument("-l", "--logs", default=True, action="store_true", help="Ignore logs directory?")
    parser.add_argument("-r", "--recursive", default=False, action="store_true", help="Remove elements recusrivly?")
    args = parser.parse_args()

    # Setup logger
    debug = args.verbose
    logger = setup_logger(debug, "Clear Logger")


    # Clear logs directory
    if args.logs:
        LOGS_PATH = os.path.join(DIR_PATH, "logs")
        logs_clear = remove_from(LOGS_PATH, args.recursive, logger)







    
