from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("hand_gesture_recog_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_np = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_np = (image_np / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_np)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Prepare the output text
    output_text = f"Class: {class_name[2:]}, Confidence: {str(np.round(confidence_score * 100))[:-2]}%"

    # Overlay the text on the image
    cv2.putText(image, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Print prediction and confidence score to the console
    print(output_text)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
