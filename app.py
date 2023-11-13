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
)


cfg = {
    "model_path": "./pretrained_models/pose_landmarker_heavy.task",
    "post_url": "https://cms.horusleg.com/api/collections/CV_Angles_Felipe_Ducau/records",
}


def main():
    st.session_state.loaded_landmarks = False
    st.session_state.computed_angles = False
    st.session_state.flexion_angle = None
    st.session_state.abduction_angle = None

    sample_image_front_path = "./sample_data/sample_front.jpg"
    sample_image_side_path = "./sample_data/sample_side.jpg"

    st.title("Prosthetics Measuring App")

    # Using st.markdown
    st.markdown("## Instructions")
    # Create an expander
    with st.expander("(1) Picture set up"):
        st.markdown(
            """
            - Choose a place with solid background and uniform illumination.
            - Use as little clothing as possible when taking the picture.
            - The picture frame should capture the entire body shape and head.
        """
        )
    with st.expander("(2) Front picture instructions"):
        sample_image_front = cv2.imread(sample_image_front_path)

        st.markdown(
            """
            - Place two identical chairs in front of the background.
            - Hold both chairs with your hands from the upper part (one on each side of your body)
            - Lift yourself with your arms suficiently so your standing leg is touching the floor but it is not sustaining your body weight.
            - Your hips should be in a resting state and you should not be tightening any of the muscles of your legs.
        """
        )
        st.image(
            cv2.cvtColor(sample_image_front, cv2.COLOR_BGR2RGB),
            caption="Sample front image",
            width=200,
        )

    with st.expander("(3) Side picture instructions"):
        sample_image_side = cv2.imread(sample_image_side_path)

        st.markdown(
            """
            - Stand against the wall facing to the side in a way that your limp is facing the camera.
            - Maintain your position by placing your hand against the background.
                    """
        )
        st.image(
            cv2.cvtColor(sample_image_side, cv2.COLOR_BGR2RGB),
            caption="Sample front image",
            width=200,
        )

    # Using st.markdown
    st.markdown("## Analysis setup")

    analysis_options = ["Abduction (front picture)", "Flexion (side picture)"]

    analysis = st.selectbox("Type of Analysis:", analysis_options, index=None)

    limp_leg = st.selectbox("Limp leg", ["RIGHT", "LEFT"], index=None)

    if analysis is not None and limp_leg is not None:
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            test_number = uploaded_file.name[2:]
            test_number = test_number.split(".")[0]
            test_number = int(test_number)

            st.image(uploaded_file, caption="Original Image", width=150)

            mp_image = load_image_data(uploaded_file, src='streamlit')

            pose_detection = pose_estimation(mp_image, cfg)
            annotated_image = draw_landmarks_on_image(
                mp_image.numpy_view()[:, :, :3], pose_detection
            )

            st.image(
                annotated_image,
                caption="Processed Image",
                use_column_width=True,
                channels="GRAY",
            )

            if len(pose_detection.pose_landmarks) == 0:
                st.error(
                    """
                    We couldn't estimate the pose from the
                    picture you uploaded.
                    Please try again with a new picture.
                """
                )

            else:
                analysis_landmarks = pose_detection.pose_landmarks[0][23:27]
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
        # Front analysis
        if analysis == analysis_options[0]:
            if st.button("Estimate Abduction Angle"):
                analysis_result = estimate_abduction(
                    pose_coordinates,
                    limp_leg
                )
                st.write(f"{analysis_result:.2f} degrees")
                flexion_angle = None
                abduction_angle = analysis_result
                st.session_state.computed_angles = True
                st.session_state.abduction_angle = abduction_angle
                st.session_state.flexion_angle = None

        elif analysis == analysis_options[1]:
            if st.button("Estimate Flexion Angle"):
                analysis_result = estimate_flexion(
                    analysis_landmarks,
                    limp_leg
                )
                st.write(f"{analysis_result:.2f} degrees")
                flexion_angle = analysis_result
                st.session_state.computed_angles = True
                st.session_state.abduction_angle = None
                st.session_state.flexion_angle = flexion_angle

        # Post to Horus API
        if st.session_state.computed_angles:
            st.markdown("Analysis JSON")
            post_data = {"data": {
                "abduction_angle": str(st.session_state.abduction_angle),
                "flexion_angle": str(st.session_state.flexion_angle),
                "test_number": 3247,
                "any_data_as_file": 'CF3247.png'
            }}
            st.write(post_data)

            if st.button("Post to API"):
                import json
                cv2.imwrite(
                    './result.png',
                    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                )

                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
                }

                files = {"data": open('./result.png', 'rb')}

                response = requests.post(
                    cfg["post_url"],
                    data=json.dumps(post_data),
                    headers=headers,
                )
                st.write(response)
                print(response)


if __name__ == "__main__":
    main()
