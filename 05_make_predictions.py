import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import serial
import time
import os

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sunglasses images
sunglasses_male = cv2.imread('images/sunglasses-black.png', cv2.IMREAD_UNCHANGED)
sunglasses_female = cv2.imread('images/sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)

male = ["images/glass5.png", "images/glass6.png", "images/glass10.png", "images/glass9.png", "images/sunglasses-black.png"]
female = ["images/glass1.png", "images/glass25.png", "images/glass17.png", "images/glass14.png", "images/sunglasses-kitty-02.png"]

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
    item_name = os.path.splitext(item_name)[0]  # Remove the file extension from the item name
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

def detect_ok_sign(hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            if (abs(thumb_tip.x - index_tip.x) < 0.02 and
                abs(thumb_tip.y - index_tip.y) < 0.02 and
                index_tip.y < index_mcp.y):
                return True
    return False

def is_thumbs_up(hand_landmarks):
    for hand_landmark in hand_landmarks:
        thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_IP]
        if thumb_tip.y < thumb_ip.y < thumb_mcp.y:
            return True
    return False

def is_thumbs_down(hand_landmarks):
    for hand_landmark in hand_landmarks:
        thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_IP]
        if thumb_tip.y > thumb_ip.y > thumb_mcp.y:
            return True
    return False

def fetch_cart_details(customer_id):
    conn = sqlite3.connect('customer_faces_data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_id,))
    customer_name = cursor.fetchone()

    cursor.execute("SELECT item_name, item_count FROM cart WHERE customer_uid = ?", (customer_id,))
    cart_items = cursor.fetchall()

    conn.close()

    return customer_name, cart_items



def main():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

    # Load Haarcascade for face detection
    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    cam = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    ok_sign_detected = False
    current_sunglass_index = 0
    last_detection_time = time.time()  # Initialize time tracking
    
    # Initialize serial communication
    ser = serial.Serial('COM6', 9600)  # Replace 'COM12' with your actual serial port
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        conf = -1  # Initialize conf to a default value
        sunglasses_name = "None"
        id_= -1
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
                sunglasses_list = male
            else:
                sunglasses_list = female
            
            sunglasses_path = sunglasses_list[current_sunglass_index]
            sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
            sunglasses_name = sunglasses_path.split("/")[-1]

            if sunglasses is None:
                print(f"Error loading sunglasses image: {sunglasses_path}")
                continue

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
        
        if results.multi_hand_landmarks:
            current_time = time.time()
            if current_time - last_detection_time >= 5.0:  # Check if 5 seconds have passed
                if detect_ok_sign(results.multi_hand_landmarks):
                    ok_sign_detected = True
                    update_ok_sign_detected(id_, 1)
                    add_item_to_cart(id_, sunglasses_name)  # Add the sunglasses to the cart

                    def format_for_lcd(customer_name, cart_items):
                        screens = []
                        
                        # First screen: Customer name
                        screens.append(f"Customer:{' ' * (16 - len('Customer:'))}\n{customer_name[:16]}")
                        
                        # Subsequent screens: Cart items
                        for item_name, item_count in cart_items:
                            item_line = f"{item_name[:13]}: {item_count}"
                            if len(screens) % 2 == 1:  # Start of a new screen
                                screens.append(item_line.ljust(16))
                            else:  # Second line of the current screen
                                screens[-1] += f"\n{item_line.ljust(16)}"
                        
                        # If the last screen has only one line, add an empty line
                        if len(screens) % 2 == 1:
                            screens[-1] += "\n" + " " * 16
                        
                        return screens

                    # Fetch cart details
                    customer_name, cart_items = fetch_cart_details(id_)

                    # Format cart details for LCD
                    lcd_screens = format_for_lcd(customer_name, cart_items)

                    # Send cart details via Serial
                    for screen in lcd_screens:
                        ser.write(screen.encode())
                        ser.write(b'\x00')  # Null terminator to separate screens
                        print(f"Sent screen:\n{screen}")

                    print("All data sent successfully via Serial")

                    # Fetch cart details
                    customer_name, cart_items = fetch_cart_details(id_)
                    format_for_lcd(customer_name, cart_items)
                    # cart_details = f"Customer: {customer_name}\nCart Items:\n"
                    # for item_name, item_count in cart_items:
                    #     cart_details += f"{item_name}: {item_count}\n"

                    # # Send cart details via Serial
                    # ser.write(cart_details.encode())
                    # print("Data sent successfully via Serial")

                    current_sunglass_index = (current_sunglass_index + 1) % len(sunglasses_list)
                    ok_sign_detected = False
                    last_detection_time = current_time  # Update last detection time
            
                if is_thumbs_down(results.multi_hand_landmarks):
                    current_sunglass_index = (current_sunglass_index + 1) % len(sunglasses_list)
                    last_detection_time = current_time  # Update last detection time
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the current sunglasses on the frame
        cv2.putText(frame, f"Current Sunglasses: {sunglasses_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
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
