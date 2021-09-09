import os, logging

if __name__ == "__main__":
    """
        Initializes all needed directories to work
        with the workspace.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SETUP")
    

    current_dir = os.path.dirname(os.path.realpath(__file__))
    workspace_path = os.path.join(current_dir, "..", "wp")
    workspace_content = os.listdir(workspace_path)

    # Create a datasets directory
    if not "datasets" in workspace_content:
        logger.info("CREATING: datasets directory")
        dataset_dir = os.path.join(workspace_path, "datasets")
        os.mkdir(dataset_dir)

    else:
        logger.info("SKIP: Datasets directory already existing")


    # Create metrics directory
    if "metrics" not in workspace_content:
        logger.info("CREATING: metrics directory")
        metrics_dir = os.path.join(workspace_path, "metrics")
        os.mkdir(metrics_dir)
    else:
        logger.info("SKIP: Metrics directory already ")




