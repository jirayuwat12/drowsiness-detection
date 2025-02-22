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

## Dataset

In this project, we have used 2 datasets:

1. [Drowsiness Detection Dataset, *Prasad V Patil*](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset/data)
2. [Driver Drowsiness Detection (DDD), *Ismail Nasri*](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

## Reference

1. [MediaPipe by Google](https://mediapipe-studio.webapps.google.com/studio/demo/face_detector)
    - We use MP to create our dataset(extract face and eye position from our video)
