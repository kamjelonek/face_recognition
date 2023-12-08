import cv2
import mediapipe as mp
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Error: Camera not accessible")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Frame not read properly")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = [
                    (mp_face_mesh.FACEMESH_LEFT_EYE, (0, 255, 0)),
                    (mp_face_mesh.FACEMESH_RIGHT_EYE, (0, 255, 0)),
                    (mp_face_mesh.FACEMESH_LEFT_EYEBROW, (0, 255, 255)),
                    (mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (0, 255, 255)),
                    (mp_face_mesh.FACEMESH_LIPS, (0, 0, 255)),
                    (mp_face_mesh.FACEMESH_FACE_OVAL, (0, 255, 0))
                ]
                for feature, color in features:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=feature,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1),
                        landmark_drawing_spec=None
                    )

                # Rysowanie źrenic
                left_pupil_point = face_landmarks.landmark[468]
                right_pupil_point = face_landmarks.landmark[473]
                left_pupil_coords = (int(left_pupil_point.x * frame.shape[1]), int(left_pupil_point.y * frame.shape[0]))
                right_pupil_coords = (int(right_pupil_point.x * frame.shape[1]), int(right_pupil_point.y * frame.shape[0]))

                cv2.circle(frame, left_pupil_coords, 3, (255, 0, 255), -1)
                cv2.circle(frame, right_pupil_coords, 3, (255, 0, 255), -1)

                # Rysowanie nosa
                indexes_nose = [4, 6, 168, 5, 195, 4, 197]
                for i in range(len(indexes_nose) - 1):
                    pt1 = face_landmarks.landmark[indexes_nose[i]]
                    pt2 = face_landmarks.landmark[indexes_nose[i + 1]]
                    x1, y1 = int(pt1.x * frame.shape[1]), int(pt1.y * frame.shape[0])
                    x2, y2 = int(pt2.x * frame.shape[1]), int(pt2.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Obliczanie obszaru twarzy dla DeepFace
                h, w, c = frame.shape
                x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
                x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
                y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                face_img = frame[y_min:y_max, x_min:x_max]
                if face_img.size != 0:
                    try:
                        # Analiza twarzy
                        analysis = DeepFace.analyze(face_img, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

                        analysis_result = analysis[0]

                        face_oval_spec = mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1)
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                                    landmark_drawing_spec=face_oval_spec,
                                    connection_drawing_spec=face_oval_spec)


                        cv2.putText(frame,
                                    f"{analysis_result['dominant_emotion']}, {analysis_result['age']}, {analysis_result['dominant_gender']}",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)
                    except Exception as e:
                        print("DeepFace Error:", e)

        cv2.imshow('VIDEO', frame)

        if cv2.waitKey(1) == 27:
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()


