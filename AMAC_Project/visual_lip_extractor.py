import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

LIP_LANDMARKS = list(range(61, 88))  # mouth region

def extract_lip_landmarks():
    cap = cv2.VideoCapture(0)

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                lip_points = []

                for idx in LIP_LANDMARKS:
                    lm = landmarks.landmark[idx]
                    lip_points.append([lm.x, lm.y])

                lip_points = np.array(lip_points)

                for x, y in lip_points:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(x*w), int(y*h)), 2, (0,255,0), -1)

            cv2.imshow("Lip Reading - Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_lip_landmarks()
