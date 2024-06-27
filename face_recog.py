import cv2
import mediapipe as mp
import face_recognition

# Initialize MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load a reference image of yourself and get the face encoding
reference_image = face_recognition.load_image_file("image.png")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Create FaceMesh object
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect one face at a time
    refine_landmarks=True,  # Refine landmarks for better accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect the face landmarks
        results = face_mesh.process(rgb_frame)

        # Draw the face mesh on the frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Convert the frame to RGB and get face encodings
        rgb_small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame for faster processing
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare face encodings with the reference encoding
            matches = face_recognition.compare_faces([reference_encoding], face_encoding)
            name = "Unknown"

            if True in matches:
                name = "You"

            # Display the name below the detected face
            for (top, right, bottom, left) in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Face Recognition with MediaPipe Face Mesh', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
