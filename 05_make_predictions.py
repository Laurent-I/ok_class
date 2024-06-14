import cv2
import numpy as np
import mediapipe as mp
import sqlite3

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sunglasses images
sunglasses_male = cv2.imread('images/sunglasses-black.png', cv2.IMREAD_UNCHANGED)
sunglasses_female = cv2.imread('images/sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)

# Load the gender detection model
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define a scaling factor for the sunglasses size
sunglasses_scale = 0.9  # Adjust as needed

# Gender classification threshold
gender_threshold = 0.6  # Adjust as needed

def create_cart_table():
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cart (
            cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_uid INTEGER,
            item_name TEXT,
            item_count INTEGER,
            FOREIGN KEY(customer_uid) REFERENCES customers(customer_uid)
        )
    ''')
    conn.commit()
    conn.close()


def add_item_to_cart(customer_id, item_name):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    
    # Check if the item is already in the cart for the customer
    cursor.execute('''
        SELECT item_count FROM cart WHERE customer_uid = ? AND item_name = ?
    ''', (customer_id, item_name))
    result = cursor.fetchone()
    
    if result:
        # Item already in cart, update the count
        new_count = result[0] + 1
        cursor.execute('''
            UPDATE cart SET item_count = ? WHERE customer_uid = ? AND item_name = ?
        ''', (new_count, customer_id, item_name))
    else:
        # Item not in cart, insert a new row
        new_count = 1
        cursor.execute('''
            INSERT INTO cart (customer_uid, item_name, item_count) VALUES (?, ?, 1)
        ''', (customer_id, item_name))
    
    conn.commit()
    conn.close()

    return new_count


def get_customer_name(predicted_id):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (predicted_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return "Unknown"

def add_ok_sign_column():
    try:
        conn = sqlite3.connect('customer_faces_data.db')
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE customers ADD COLUMN ok_sign_detected INTEGER DEFAULT 0")
        conn.commit()
        print("Column 'ok_sign_detected' added successfully.")
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")

def update_ok_sign_detected(predicted_id, ok_sign_detected):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE customers SET ok_sign_detected = ? WHERE customer_uid = ?", (ok_sign_detected, predicted_id))
    conn.commit()
    conn.close()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_ok_sign(image, hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            if (abs(thumb_tip.x - index_tip.x) < 0.03 and
                abs(thumb_tip.y - index_tip.y) < 0.03 and
                index_tip.y < index_mcp.y):
                return True
    return False

def main():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

    # Load Haarcascade for face detection
    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    ok_sign_count = 0
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        conf = -1  # Initialize conf to a default value
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = faceRecognizer.predict(roi_gray)
            
            if conf >= 45:
                customer_name = get_customer_name(id_)
                label = f"{customer_name} - {round(conf, 2)}%"
            else:
                label = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Prepare the input image for gender detection
            face_roi = frame[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Run gender detection
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()

            # Get the predicted gender and confidence
            gender = "Male" if genderPreds[0, 0] > gender_threshold else "Female"
            gender_confidence = genderPreds[0, 0]

            # Choose the sunglasses based on gender
            if gender == "Male":
                sunglasses = sunglasses_male
                sunglasses_name = 'Black Sunglasses'
            else:
                sunglasses = sunglasses_female
                sunglasses_name = 'Kitty Sunglasses'

            # Calculate the position and size of the sunglasses
            sunglasses_width = int(sunglasses_scale * w)
            sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])

            # Resize the sunglasses image
            sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

            # Calculate the position to place the sunglasses
            x1 = x + int(w / 2) - int(sunglasses_width / 2)
            x2 = x1 + sunglasses_width
            y1 = y + int(0.35 * h) - int(sunglasses_height / 2)
            y2 = y1 + sunglasses_height

            # Adjust for out-of-bounds positions
            x1 = max(x1, 0)
            x2 = min(x2, frame.shape[1])
            y1 = max(y1, 0)
            y2 = min(y2, frame.shape[0])

            # Create a mask for the sunglasses
            sunglasses_mask = sunglasses_resized[:, :, 3] / 255.0
            frame_roi = frame[y1:y2, x1:x2]

            # Blend the sunglasses with the frame
            for c in range(0, 3):
                frame_roi[:, :, c] = (1.0 - sunglasses_mask) * frame_roi[:, :, c] + sunglasses_mask * sunglasses_resized[:, :, c]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        ok_sign_detected = detect_ok_sign(rgb_frame, results.multi_hand_landmarks)
        
        if ok_sign_detected:
            cv2.putText(frame, "OK Sign Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if conf >= 45:
                update_ok_sign_detected(id_, 1)
                ok_sign_count = add_item_to_cart(id_, sunglasses_name)  # Add the sunglasses to the cart and get the count
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the OK sign count on the frame
        cv2.putText(frame, f"Items in Cart: {ok_sign_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        cv2.imshow('Face and Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Uncomment the next line when running for the first time to add the column
    # add_ok_sign_column()
    create_cart_table()  # Create the cart table if it doesn't exist
    main()


