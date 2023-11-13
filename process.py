import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometer import Line, Point, angle
import cv2
import numpy as np

import os
import copy

import math

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def pose_estimation(rgb_image, cfg):
    # Create Detector
    base_options = python.BaseOptions(model_asset_path=cfg["model_path"])
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # detect
    detection_result = detector.detect(rgb_image)
    return detection_result


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Overlay detection results on top of RGB image.
    Highlights hips and knees.

    :param rgb_image: <np.array> input RGB image
    :detection_result: <PoseLandmarkerResult> as outcome from
                       vision.PoseLandmarker detector.
    :returns: annotated image
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        styles = solutions.drawing_styles.get_default_pose_landmarks_style()
        for k in styles.keys():
            if k.real in [26, 25, 24, 23]:
                st = copy.deepcopy(styles[k])
                st.thickness = 15
                st.color = (210, 97, 255)
                styles[k] = st

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            styles,
        )
    return annotated_image


def cv2_imshow(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def load_image_data(uploaded_file):
    image_data = uploaded_file.read()

    # Convert the bytes to a NumPy array
    np_image = np.frombuffer(image_data, dtype=np.uint8)

    # Use OpenCV to decode the NumPy array into an image
    # Use 1 to load color image, 0 for grayscale
    opencv_image = cv2.imdecode(np_image, 1)
    mp_image = mp.Image(
        data=cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB),
        image_format=mp.ImageFormat.SRGB,
    )
    return mp_image


def get_coordinates(pose_landmarks):
    """
    Get x, y coordinates from list of
    NormalizedLandmarks
    """
    coordinates = []
    for pl in pose_landmarks:
        coordinates.append((pl.x, pl.y))
    return coordinates


def center_vector(vector):
    rv = vector
    rv[:, 0] -= rv[0, 0]
    rv[:, 1] -= rv[0, 1]

    return rv


def find_perpendicular_vector(point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    perpendicular_vector = (y1 - y0, -(x1 - x0))
    return perpendicular_vector


def estimate_abduction(pose_coordinates, limp_leg):
    """
    :param pose_landmarks: <list>
        List of NormalizedLandmark objects.
        corresponding to:
        [left_hip, right_hip, left_knee, right_knee]

    :param limp_leg: <str> 'RIGHT'|'LEFT'
    :returns: <float> abduction angle in degrees.
    """
    pose_points = [Point(*pc) for pc in pose_coordinates]
    hip_vector = Line(pose_points[0], pose_points[1])

    # Compute the vertical reference
    # perpendicular to hip line
    if limp_leg == "LEFT":
        leg_vector = Line(pose_points[0], pose_points[2])
        vertical_reference = hip_vector.perpendicular(through=pose_points[0])
    else:
        leg_vector = Line(pose_points[1], pose_points[3])
        vertical_reference = hip_vector.perpendicular(through=pose_points[1])

    # Make sure that the vertical reference is pointing down
    if vertical_reference.direction[1] < 0:
        vertical_reference = Line(-vertical_reference)

    # Compute abduction angle
    if vertical_reference.direction == leg_vector.direction:
        abduction_dgr = 0
    elif limp_leg == "LEFT":
        abduction_dgr = angle(vertical_reference, leg_vector) * 180 / math.pi
    else:
        abduction_dgr = -angle(vertical_reference, leg_vector) * 180 / math.pi

    return abduction_dgr


def estimate_flexion(pose_coordinates, limp_leg):
    """
    :param pose_landmarks: <list>
        List of NormalizedLandmark objects.
        corresponding to:
        [left_hip, right_hip, left_knee, right_knee]

    :param limp_leg: <str> 'RIGHT'|'LEFT'
    :returns: <float> flexion angle in degrees.
    """
    pose_points = [Point(*pc) for pc in pose_coordinates]

    vertical_line = Line(Point(0, 0), Point(0, 1))

    # Compute the vertical reference
    # perpendicular to hip line
    if limp_leg == "LEFT":
        leg_vector = Line(pose_points[0], pose_points[2])
        vertical_reference = vertical_line.parallel(through=pose_points[0])
    else:
        leg_vector = Line(pose_points[1], pose_points[3])
        vertical_reference = vertical_line.parallel(through=pose_points[1])

    # Make sure that the vertical reference is pointing down
    if vertical_reference.direction[1] < 0:
        vertical_reference = Line(-vertical_reference)

    # Compute flexoin angle
    if vertical_reference.direction == leg_vector.direction:
        flexion_dgr = 0
    elif limp_leg == "LEFT":
        flexion_dgr = angle(leg_vector, vertical_reference) * 180 / math.pi
    else:
        flexion_dgr = angle(vertical_reference, leg_vector) * 180 / math.pi

    return flexion_dgr
