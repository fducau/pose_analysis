import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometer import Line, Point, angle
import cv2
import numpy as np

import math
from typing import List, Mapping, Optional, Tuple, Union

from mediapipe.python.solutions.drawing_utils import (
    DrawingSpec,
    WHITE_COLOR,
    BLACK_COLOR,
    RED_COLOR,
    GREEN_COLOR,
    BLUE_COLOR,
    _BGR_CHANNELS,
    _normalized_to_pixel_coordinates,
)


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 0.7
font_color = (255, 0, 0)

SKIP_JOINT_ANNOTATIONS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,    # Face
    17, 18, 19, 20, 21, 22,    # Hands
    29, 30, 31, 32    # Feet
]


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
    pixel_scale: Union[None, float] = None
):
    # Modified from mediapipe
    # (https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py)
    """
    Draws the landmarks and the connections on the image.

      Args:
        image: A three channel BGR image represented as numpy ndarray.
        landmark_list: A normalized landmark list proto message to be annotated on
          the image.
        connections: A list of landmark index tuples that specifies how landmarks to
          be connected in the drawing.
        landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
          landmarks to the DrawingSpecs that specifies the landmarks' drawing
          settings such as color, line thickness, and circle radius. If this
          argument is explicitly set to None, no landmarks will be drawn.
        connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
          connections to the DrawingSpecs that specifies the connections' drawing
          settings such as color and line thickness. If this argument is explicitly
          set to None, no landmark connections will be drawn.
        is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
          landmarks, only contours will be drawed.

      Raises:
        ValueError: If one of the followings:
          a) If the input image is not three channel BGR.
          b) If any connetions contain invalid landmark index.
      """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue

        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                f'from landmark #{start_idx} to landmark #{end_idx}.')

            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                

                cv2.line(image, idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx], drawing_spec.color,
                        drawing_spec.thickness)
                if pixel_scale is not None:
                    if start_idx not in SKIP_JOINT_ANNOTATIONS and end_idx not in SKIP_JOINT_ANNOTATIONS:

                        mid_pt = get_mid_point(idx_to_coordinates[start_idx],
                                               idx_to_coordinates[end_idx])
                        line_len = point_length(idx_to_coordinates[start_idx],
                                                idx_to_coordinates[end_idx])
                        line_len_cm = pixel_scale * line_len

                        cv2.putText(
                            image,
                            f"{int(line_len_cm)}cm",
                            mid_pt,
                            font,
                            font_scale,
                            font_color
                        )

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                    drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                    drawing_spec.color, drawing_spec.thickness)


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


def draw_landmarks_on_image(
    rgb_image,
    detection_result,
    pixel_scale=None
):
    """
    Overlay detection results on top of RGB image.
    Highlights hips and knees.

    :param rgb_image: <np.array> input RGB image
    :detection_result: <PoseLandmarkerResult> as outcome from
                       vision.PoseLandmarker detector.
    :pixel_scale: <float> Optional. Reference centimeters per pixel.
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

        draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            styles,
            pixel_scale=pixel_scale,
        )
    return annotated_image


def cv2_imshow(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def load_image_data(
    uploaded_file,
    src='streamlit',
    height=None,
):
    """
    Load image data to process with mediapipe.
    :param uploaded_file: image object or image path.
    :src: <str> 'streamlit'|'file'. If coming from streamlit
          it treates it as an uploaded file, otherwise
          it opens it from disk.
    :height: rezises to given image height.
    :returns: mediapipe image.
    """
    if src not in ['streamlit', 'file']:
        raise ValueError('Invalid input image')

    if src == 'streamlit':
        image_data = uploaded_file.read()
    else:
        image_data = open(uploaded_file, 'rb').read()

    # Convert the bytes to a NumPy array
    np_image = np.frombuffer(image_data, dtype=np.uint8)

    # Use OpenCV to decode the NumPy array into an image
    # Use 1 to load color image, 0 for grayscale
    opencv_image = cv2.imdecode(np_image, 1)

    if height is not None:
        aspect_ratio = 1.0 * opencv_image.shape[1] / opencv_image.shape[0]
        new_height = height
        new_width = int(new_height * aspect_ratio)
        opencv_image = cv2.resize(opencv_image, (new_width, new_height))

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


def get_top_head_y(segmentation_mask):
    # Get (x, y) coordinates for the top of the head
    if len(segmentation_mask.shape) == 2:
        y, x = np.where(segmentation_mask > 0.6)
    else:
        y, x, z = np.where(segmentation_mask > 0.6)

    min_y = np.min(y)

    img_height = segmentation_mask.shape[0]

    min_y_relative = 1.0 * min_y / img_height
    return min_y_relative


def get_heels_y(pose_coordinates):
    """
    Get the mean y point for a line that connects
    both heels.

    :param  pose_coordinates: <list>
        List of NormalizedLandmark objects.
    """
    left_heel = pose_coordinates[29][1]
    right_heel = pose_coordinates[30][1]

    min_heel = min(left_heel, right_heel)
    mid_point = abs(left_heel - right_heel) / 2 + min_heel

    return mid_point


def get_pixel_scale(
    segmentation_mask,
    height,
    heels_y,
    top_head_y
):
    pixels_y = segmentation_mask.shape[0]

    heels_y_pix = pixels_y * heels_y
    top_head_y_pix = pixels_y * top_head_y

    height_pixels = abs(heels_y_pix - top_head_y_pix)
    h_per_pixel = height / height_pixels

    return h_per_pixel


def get_mid_point(a, b):
    """
    Get the mean point between points
    a and b

    :param a: <tuple> (x1, y1)
    :param b: <tuple> (x2, y2)
    :returns: <tuple> (xmid, ymid)
    """

    xmin = min(a[0], b[0])
    ymin = min(a[1], b[1])

    xdiff = abs(a[0] - b[0])
    ydiff = abs(a[1] - b[1])

    return (int(xmin + xdiff / 2), int(ymin + ydiff / 2))



def point_length(a, b):
    """
    Point length between points a and b
    :param a: <tuple> (x1, y1)
    :param b: <tuple> (x2, y2)
    :returns: <int|float> point length
    """

    xmin = min(a[0], b[0])
    ymin = min(a[1], b[1])

    xdiff = abs(a[0] - b[0])
    ydiff = abs(a[1] - b[1])

    return math.sqrt(xdiff * xdiff + ydiff * ydiff)
