import argparse
import logging
import os

import yaml


def main(mode: str, config: dict, logger: logging.Logger) -> None:
    """
    The main function to run the drowsiness detection

    :param mode: The mode to run the drowsiness detection in (video or webcam)
    :type mode: str
    :param config: The configuration file
    :type config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if mode == "video":
        logger.info("Running the drowsiness detection in video mode")
        # TODO: Implement the video mode
    elif mode == "webcam":
        logger.info("Running the drowsiness detection in webcam mode")
        # TODO: Implement the webcam mode
    else:
        logger.error('The mode should be either "video" or "webcam"')
        raise ValueError('The mode should be either "video" or "webcam"')

    logger.info(f"Exiting the drowsiness detection")


if __name__ == "__main__":
    # Create a parser object
    parser = argparse.ArgumentParser(description="Run drowsiness detection on a video")

    parser.add_argument("--mode", type=str, help="The mode to run the drowsiness detection in (video or webcam)")
    parser.add_argument("--log_folder", type=str, default="./drowsiness_logs", help="The path to the log file")
    parser.add_argument("--config", type=str, required=True, help="The path to the configuration file")

    # Parse the arguments
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Create a logger object
    os.makedirs(args.log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_folder, "drowsiness_detection.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    stdout_handler = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(stdout_handler)

    # Run the main function
    main(args.mode, config, logger)
