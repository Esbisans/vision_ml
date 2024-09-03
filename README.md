# Vision Computer machine learning project

This project contains a Python-based computer vision project utilizing live video input from a camera device. The project is built using the MediaPipe machine learning library and OpenCV for video capture and processing. It consists of three main sections: Object Detection, Hand Tracking, and Face Landmark Detection.


## Overview
 
This project showcases real-time computer vision capabilities by leveraging machine learning techniques. It is divided into three key sections:

- **Object Detection:** Identifies and highlights objects in a live video stream using the COCO dataset.
- **Hand Tracking:** Detects and tracks hand movements and gestures using landmark detection.
- **Face Landmark Detection:** Maps and identifies key facial landmarks in real-time.

## Object Detection

The object detection module uses the [COCO dataset](https://cocodataset.org/#home) to identify and classify objects in the video feed. The model is pre-trained and can recognize a variety of common objects such as people, vehicles, and household items. The detected objects are highlighted with bounding boxes and labeled with their respective class names.

[label list](https://github.com/Esbisans/vision_ml/blob/main/coco_labels.txt)

### Demo 
![Demo](assets/object_detection.gif)


## Hand Tracking

In the hand tracking section, the project uses MediaPipe's hand landmarks to accurately track hand movements and gestures. This feature is particularly useful for gesture recognition applications, allowing for the detection of specific hand poses in real-time.

### Gestures

| Gesture Name  | Image                                          |
|---------------|------------------------------------------------|
| Closed_Fist   | ![Closed_Fist](assets/Closed_Fist.jpg) |
| Open_Palm     | ![Open_Palm](assets/Open_Palm.jpg)     |
| Pointing_Up   | ![Pointing_Up](assets/Pointing_Up.jpg) |
| Thumb_Down    | ![Thumb_Down](assets/Thumb_Down.jpg)    |
| Thumb_Up      | ![Thumb_Up](assets/Thumb_Up.jpg)      |
| Victory       | ![Victory](assets/Victory.jpg)       |
| ILoveYou      | ![ILoveYou](assets/ILoveYou.jpg)      |

### Demo 
<img src="assets/hand_tracking.gif" alt="Demo" width="450"/>



## Face Landmark Detection

The face landmark detection module identifies key points on a person's face, such as the eyes, nose, and mouth, using MediaPipe's facial landmark model. This is important for detect facial expression, analysis of face alignment, and more.


## Project's Link 

[visionml.streamlit.app](https://visionml.streamlit.app/)

(It might go into sleep mode due to inactivity because of Streamlit's free tier.)