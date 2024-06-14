# Face Recognition and Hand Gesture Detection Project

This project is a comprehensive system that combines face recognition and hand gesture detection using computer vision techniques. The system is built using Python and leverages various libraries such as OpenCV, Tensorflow, MediaPipe, and SQLite.

## Features

- Capture and store face images for multiple customers in a SQLite database.
- Cluster face images using K-Means clustering and the VGG16 pre-trained model for feature extraction.
- Train an LBPH (Local Binary Patterns Histograms) face recognizer model using the clustered face images.
- Detect faces in real-time video feed using the trained LBPH model.
- Display the recognized customer's name and confidence score.
- Detect the "OK" hand gesture using MediaPipe's hand landmark detection.
- Update the database to mark when a customer has shown the "OK" sign.
- Allow the user to virtually try on sunglasses when their face is recognized.
- Add the selected sunglasses to the user's cart in the database when they make the "OK" sign gesture.

## Project Structure

```
.
├── dataset/                  # Directory to store face images
├── models/                   # Directory to store pre-trained models
│   ├── haarcascade_frontalface_default.xml
├── 01_create_dataset.py      # Script to capture face images and store in database
├── 02_create_clusters.py     # Script to cluster face images using K-Means and VGG16
├── 03_rearrange_data.py      # Script to rearrange data after clustering
├── 04_train_model.py         # Script to train the LBPH face recognizer model
├── 05_make_predictions.py    # Script to run face recognition and hand gesture detection
├── customer_faces_data.db    # SQLite database to store customer information and face images
└── README.md
```

## Setup

1. Install the required Python packages:

```
pip install opencv-python numpy tensorflow keras scikit-learn mediapipe tqdm
```

2. Download the pre-trained `haarcascade_frontalface_default.xml` model from the OpenCV repository and place it in the `models` directory.

## Usage

1. **Create Dataset**: Run `01_create_dataset.py` to capture face images for each customer and store them in the `dataset` directory and the SQLite database.

2. **Create Clusters**: Run `02_create_clusters.py` to cluster the face images using K-Means clustering and VGG16 feature extraction. This will create a `dataset-clusters` directory with the clustered images.

3. **Rearrange Data**: Run `03_rearrange_data.py` to rearrange the data after clustering. This will move the clustered images back to the `dataset` directory and remove any invalid entries from the database.

4. **Train Model**: Run `04_train_model.py` to train the LBPH face recognizer model using the face images in the `dataset` directory. The trained model will be saved as `trained_lbph_face_recognizer_model.yml` in the `models` directory.

5. **Make Predictions**: Run `05_make_predictions.py` to start the real-time face recognition and hand gesture detection system. The script will open a webcam feed and display the recognized customer's name, confidence score, and detect the "OK" hand gesture using mediapipe. When the "OK" sign is detected for a recognized customer, the database will be updated accordingly. Allow the user to virtually try on sunglasses when their face is recognized.
Add the selected sunglasses to the user's cart in the database when they make the "OK" sign gesture.

## Notes

- The project is designed to work with a local webcam. If you want to use a different video source, modify the `cv2.VideoCapture` call in `05_make_predictions.py`.
- The database file `customer_faces_data.db` will be created automatically if it doesn't exist.
- The `models` directory should contain the `haarcascade_frontalface_default.xml` file for face detection. You can download it from the OpenCV repository.
- The number of clusters for K-Means clustering can be adjusted in `02_create_clusters.py`.
- The confidence threshold for face recognition can be adjusted in `05_make_predictions.py`.
- You need to uncomment `add_ok_sign_column()` in the `05_make_predictions.py` if you are running it for the first time
- You will need `tensorflow version 2.12.0` and `keras version 2.12.0` with a python version of `3.11.9`

## License

This project is licensed under the [MIT License](LICENSE).