import cv2
import mediapipe as mp

# Inicializar mediapipe
mp_malla_rostro = mp.solutions.face_mesh
mp_dibujo = mp.solutions.drawing_utils

malla_rostro = mp_malla_rostro.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=9,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

 
# Captura de cámara
captura = cv2.VideoCapture(0)

while captura.isOpened():
    exito, fotograma = captura.read()
    if not exito:
        print("⚠ No se pudo acceder a la cámara")
        break

    fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
    resultados = malla_rostro.process(fotograma_rgb)

    if resultados.multi_face_landmarks:
        for puntos_rostro in resultados.multi_face_landmarks:
            mp_dibujo.draw_landmarks(
                fotograma,
                puntos_rostro,
                mp_malla_rostro.FACEMESH_TESSELATION,
                mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    cv2.imshow("Reconocimiento Facial (test camara)", fotograma)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

captura.release()
cv2.destroyAllWindows()
