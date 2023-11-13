# Pose Analysis Demo

Estimate flexion and abduction angles of leg amputees from pictures.

Based on `mediapipe`


# Run
  - Clone the repo
  - Download model weights from [here](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) and save them under `./pretrained_models/`
  - Create virtual envirnoment `python3 -m venv env`
  - Activate virtual environment `source env/bin/activate`
  - Install requirements `pip install -r requirements.txt`
  - Run app: `streamlit run app.py` or `make run`
