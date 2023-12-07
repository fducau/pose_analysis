import mediapipe as mp

import streamlit as st
import cv2
import numpy as np
import requests

from process import (
    draw_landmarks_on_image,
    pose_estimation,
    load_image_data,
    estimate_abduction,
    estimate_flexion,
    get_coordinates,
    get_top_head_y,
    get_heels_y,
    get_pixel_scale,
)


cfg = {
    "model_path": "./pretrained_models/pose_landmarker_heavy.task",
}
IMG_HEIGHT = 1000

def main():
    st.session_state.loaded_landmarks = False
    st.session_state.computed_angles = False
    st.session_state.flexion_angle = None
    st.session_state.abduction_angle = None

    sample_image_front_path = "./sample_data/sample_body.jpg"

    st.title("Body Measuring App")

    # Using st.markdown
    st.markdown("## Instructions")
    # Create an expander
    with st.expander("(1) Picture set up"):
        st.markdown(
            """
            - Choose a place with solid background and uniform illumination.
            - The picture frame should capture the entire body shape and head.
        """
        )
    with st.expander("(2) Pose instructions"):
        sample_image_front = cv2.imread(sample_image_front_path)

        st.markdown(
            """
            - Stand upright against the wall facing the camera.
        """
        )
        st.image(
            cv2.cvtColor(sample_image_front, cv2.COLOR_BGR2RGB),
            caption="Sample front image",
            width=200,
        )

    # Using st.markdown
    st.markdown("## Analysis setup")

    user_height = st.number_input("Please enter your height in centimeters")

    if user_height is not None:
        uploaded_file = st.file_uploader(
            "Please upload your picture according to the instructions",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Original Image", width=150)

            # Analyze image and get pose landmarks
            mp_image = load_image_data(
                uploaded_file,
                src='streamlit',
                height=IMG_HEIGHT
            )

            pose_detection = pose_estimation(mp_image, cfg)

            if len(pose_detection.pose_landmarks) == 0:
                st.error(
                    """
                    We couldn't estimate the pose from the
                    picture you uploaded.
                    Please try again with a new picture.
                """
                )

            else:
                annotated_image = draw_landmarks_on_image(
                    mp_image.numpy_view()[:, :, :3], pose_detection,
                )

                st.image(
                    annotated_image,
                    caption="Processed Image",
                    use_column_width=True,
                    channels="GRAY",
                )

                analysis_landmarks = pose_detection.pose_landmarks[0]
                segmentation_mask = pose_detection.segmentation_masks[0].numpy_view()

                pose_coordinates = get_coordinates(analysis_landmarks)
                st.success(
                    """
                    Pose successfully found.
                    Please review image annotations.
                    If the hips and knees are not properly
                    detected, please try again with a new picture.
                """
                )
                st.session_state.loaded_landmarks = True

    if st.session_state.loaded_landmarks:
        # Measurement analysis
        if st.button("Estimate Body Measures"):

            top_y_coordinate = get_top_head_y(segmentation_mask)
            bottom_y_coordinate = get_heels_y(pose_coordinates)

            h_per_pixel = get_pixel_scale(
                segmentation_mask,
                user_height,
                bottom_y_coordinate,
                top_y_coordinate
            )

            annotated_image = draw_landmarks_on_image(
                mp_image.numpy_view()[:, :, :3] * 0, pose_detection,
                pixel_scale=h_per_pixel
            )

            st.image(
                annotated_image,
                caption="Processed Image",
                use_column_width=True,
                channels="GRAY",
            )


if __name__ == "__main__":
    main()
