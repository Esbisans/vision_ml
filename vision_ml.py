import streamlit as st
import cv2 
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from streamlit_option_menu import option_menu


def visualize(image, detection_result) -> np.ndarray:

    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR_red = (255, 0, 0)  # red
    TEXT_COLOR = (255, 255, 255)  # blanco
    RECTANGLE_COLOR = (0, 0, 255)  # rojo para el rectángulo
    FONT = cv2.FONT_HERSHEY_DUPLEX

    for detection in detection_result.detections:
        # Dibuja el cuadro delimitador
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(image, start_point, end_point, RECTANGLE_COLOR, 3)


        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"

        # Dibuja el rectángulo para el texto
        text_size, _ = cv2.getTextSize(result_text, FONT, FONT_SIZE, FONT_THICKNESS)
        text_width = text_size[0] + 2 * MARGIN
        text_height = text_size[1] + 2 * MARGIN
        rect_start_point = start_point
        rect_end_point = (start_point[0] + text_width, start_point[1] + text_height)
        cv2.rectangle(image, rect_start_point, rect_end_point, RECTANGLE_COLOR, -1)  # Relleno blanco

        # Dibuja la etiqueta y la puntuación

        text_location = (start_point[0] + MARGIN, start_point[1] + MARGIN + text_size[1])
        cv2.putText(image, result_text, text_location, FONT,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def object_detection(camera_index):

    model_path = 'models/efficientdet_lite0.tflite'

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.subheader("camera not available")
        st.caption("Please change the camera channel and try again")
        ok_button = st.button("Ok")
        return


    stop_button_pressed = st.button("Stop camera", type="primary")
    frame_placeholder = st.empty()

    while cap.isOpened() and not stop_button_pressed:
        success, frame = cap.read()
        if not success:
            st.write("Video Capture Ended")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        #show image at streamlit
        frame_placeholder.image(annotated_image,channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.subheader("Thanks for using this app!")

def gesture_recognition(camera_index):

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    model_path = 'models/gesture_recognizer.task'

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.subheader("camera not available")
        st.caption("Please change the camera channel and try again")
        ok_button = st.button("Ok")
        return

    stop_button_pressed = st.button("Stop camera", type="primary")
    frame_placeholder = st.empty()

    gesture_placeholder = st.empty()

    while cap.isOpened() and not stop_button_pressed:

        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        recognition_result = recognizer.recognize(mp_image)

        annotated_image = np.copy(mp_image.numpy_view())
    
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
        else:
            top_gesture = None
        multi_hand_landmarks = recognition_result.hand_landmarks

        for hand_landmarks in multi_hand_landmarks:

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        #rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        #cv2.imshow('image', rgb_annotated_image)
        frame_placeholder.image(annotated_image,channels="RGB")
        if top_gesture:
            gesture_placeholder.subheader(f"{top_gesture.category_name}: {top_gesture.score:.2f}" )
        else:
            gesture_placeholder.subheader("No gesture detected")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    st.subheader("Thanks for using this app!")

def face_landmarks(camera_index):

    model_path = 'models/face_landmarker.task'

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.subheader("camera not available")
        st.caption("Please change the camera channel and try again")
        ok_button = st.button("Ok")
        return


    stop_button_pressed = st.button("Stop camera", type="primary")
    frame_placeholder = st.empty()

    while cap.isOpened() and not stop_button_pressed:
        success, frame = cap.read()
        if not success:
            st.write("Video Capture Ended")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = draw_landmarks_on_image(image_copy, detection_result)

        frame_placeholder.image(annotated_image,channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.subheader("Thanks for using this app!")




def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Vision ML Projects")
    st.caption("Allow camera access to use this app.")


    col1, col2 = st.columns(2) 

    with col1:
        title_project = st.empty()
        open_camera = st.empty()
        open_camera_pressed = open_camera.button("Open Camera", type="primary")

    with col2:
        camera_channel = st.empty()
        camera_index = camera_channel.selectbox("Switch camera channel", [1, 2, 3], index=0)



    with st.sidebar:

        selected = option_menu(
            menu_title="Select a project",
            options=["Object Detection", "Hand Tracking", "Face Landmarkers"],
        )


    if selected == "Object Detection":
        title_project.subheader("Object Detection")
        if open_camera_pressed:
            open_camera.empty()
            camera_channel.selectbox("Switch camera channel", [1, 2, 3], index=camera_index-1, disabled=True, key="camera_channel_disabled")
            object_detection(camera_index-1)
    if selected == "Hand Tracking":
        title_project.subheader("Hand Tracking")
        if open_camera_pressed:
            open_camera.empty()
            camera_channel.selectbox("Switch camera channel", [1, 2, 3], index=camera_index-1, disabled=True, key="camera_channel_disabled")
            gesture_recognition(camera_index-1)
    if selected == "Face Landmarkers":
        title_project.subheader("Face Landmarkers")
        if open_camera_pressed:
            open_camera.empty()
            camera_channel.selectbox("Switch camera channel", [1, 2, 3], index=camera_index-1, disabled=True, key="camera_channel_disabled")
            face_landmarks(camera_index-1)





        

if __name__ == "__main__":
    main()
