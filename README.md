# drowsiness-detection

## Installation

1. Clone the repository
2. Change the directory to the repository
```bash
cd drowsiness-detection
```
3. Install the modules
```bash
pip install .
```
> or if you need to install the modules in development mode
> ```bash
> pip install -e ".[dev]"
> ```

## Usage

1. Run the script for the video mode
    > This script will run the drowsiness detection on the video file
    respectively to the configuration file. 
```bash
python ./scripts/run_drowsiness_detection.py --mode video\
                                             --config ./path/to/config.yaml\
                                             --log_folder ./path/for/log_folder
```