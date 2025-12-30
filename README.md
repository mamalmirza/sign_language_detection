# Sign Language Detection

This project implements a Sign Language Detection system using computer vision and machine learning techniques. It utilizes **MediaPipe** for hand landmark detection and a **Random Forest Classifier** from scikit-learn for recognizing hand signs.

## Project Structure

*   `collect_imgs.py`: Script to collect image data for training. It captures frames from the webcam and saves them into class-specific folders.
*   `create_dataset.py`: Processes the collected images to extract hand landmarks using MediaPipe and saves the dataset into a pickle file (`data.pickle`).
*   `train_classifier.py`: Trains a Random Forest Classifier on the processed dataset and saves the trained model (`model.p`).
*   `inference_classifier.py`: Real-time inference script that uses the trained model to predict sign language characters from the webcam feed.

## Prerequisites

*   Python 3.x
*   OpenCV (`opencv-python`)
*   MediaPipe (`mediapipe`)
*   Scikit-learn (`sklearn`)
*   Matplotlib (`matplotlib`)
*   Numpy (`numpy`)

You can install the dependencies using pip:

```bash
pip install opencv-python mediapipe scikit-learn matplotlib numpy
```

## Usage

### 1. Data Collection

Run `collect_imgs.py` to collect training images.
The script will prompt you to collect data for each class (currently configured for classes 0, 1, and 2).
Press 'Q' to start collection for a class.

```bash
python collect_imgs.py
```

### 2. Dataset Creation

Run `create_dataset.py` to process the images and create the dataset file (`data.pickle`).

```bash
python create_dataset.py
```

### 3. Model Training

Run `train_classifier.py` to train the Random Forest model. The trained model will be saved as `model.p`.

```bash
python train_classifier.py
```

### 4. Real-time Inference

Run `inference_classifier.py` to start the real-time detection.
The script will open your webcam and display the predicted character.

```bash
python inference_classifier.py
```

## Classes

The current model is trained to recognize the following classes (mapped in `inference_classifier.py`):
*   0: 'A'
*   1: 'B'
*   2: 'L'

## Customization

You can modify `collect_imgs.py` to add more classes or change the dataset size. Remember to update the label mapping in `inference_classifier.py` if you add new classes.
