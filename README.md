<div align="center">

# Haru 2.0 Telepresence System

**An expressive, gaze-aware ROS 2 telepresence pipeline for the Haru 2.0 social robot.**

</div>

---

## Overview

Haru 2.0 is a social desktop robot. This project turns it into a **face-aware
telepresence avatar** that mirrors a remote participant's head pose, gaze,
and facial expressions in real time, while keeping the robot's eye motions
biologically plausible.

The system is designed for **video-call telepresence**: it captures a region
of the screen showing a remote participant in a video-conferencing app
(Zoom, Google Meet, Microsoft Teams, etc.) and drives the robot to mirror
that participant.

## Highlights

- **MediaPipe FaceMesh + iris tracking** — 478 landmarks at 30 FPS, giving
  per-eye iris position for gaze estimation.
- **Physics-guided LSTM (PG-LSTM) for VOR** — a small LSTM trained against a
  dual-pathway vestibulo-ocular-reflex ODE (semicircular-canal + velocity-
  storage dynamics, Ewald asymmetry, softsign saturation) keeps the eyes
  counter-rotating against head motion, like a real vestibular system.
- **Fixational micromovement simulation** — drift, tremor, and microsaccades
  are added on top of the iris signal to avoid an unnaturally static gaze.
- **GCN-based expression classifier** — an adaptive-adjacency graph
  convolutional network classifies the current normalized landmark frame into
  one of 17 robot routines (smile, surprise, thinking, etc.).
- **Head-pose calibration** — short on-startup calibration removes per-user
  baseline offsets so the robot doesn't drift to one side.

## Installation

Requirements:

- **ROS 2** with the `haru2_core_msgs` package available on your workspace
  (the node publishes motor / eye / LED commands through it).
- **Python 3.10+** (the code uses the `X | None` type-union syntax).
- Python dependencies: `pip install -r requirements.txt`
  (core: `torch`, `mediapipe`, `opencv-python`, `numpy`; training/validation:
  `matplotlib`, `scikit-learn`; teleconference capture: `mss`, `pyautogui`;
  speech: your chosen recognizer backend).

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Project Structure

```
.
├── main.py                          # Entry point (mode selector)
├── constants.py                     # All numeric constants
├── audio_manager.py                 # Audio I/O for speech recognition
├── speech_recognizer.py             # Whisper-based recognizer
├── base_system.py                   # Webcam-mode integrated node (base class)
├── teleconference_system.py         # Teleconference variant (subclass)
├── tracking.py                      # Improved tracking + calibration
├── face_eye_extractor.py            # MediaPipe + iris + micromovements
├── eye_tracker.py                   # Eye motor control + VOR coupling
├── robot_control.py                 # Motor command publishers
├── expression_manager.py            # Routine state + frame buffer
├── representative_keypoints.py      # 468 → 226 landmark down-sampling
├── simplified_landmark_plus.py      # Landmark index tables
│
├── expression_model/
│   ├── __init__.py
│   └── facial_expression_gcn.py     # GCN classifier
│
├── vor_pinn.py                      # VOR PG-LSTM runtime
├── vor_pinn_train.py                # VOR PG-LSTM training + CLI
│
├── train_haru_inverse.py            # GCN training pipeline
├── collect_data.py                  # Data collection
├── data/                            # Captured datasets (.npz)
├── models/                          # Trained checkpoints
├── plots/                           # Training plots / confusion matrices
└── lstm_results/                    # VOR validation outputs
```

## Usage

### Teleconference mode

```bash
python main.py teleconference
```

On startup, you will be prompted to mark the screen region containing the
remote participant's video tile:

1. Open your video-conferencing app and pin the participant whose face
   should drive the robot.
2. Run the command above.
3. When prompted, move the cursor to the **top-left** corner of that
   participant's video tile and wait for the countdown.
4. Move the cursor to the **bottom-right** corner and wait for the
   countdown.
5. The robot will now run a short head-pose calibration (~30 frames of the
   captured face). Hold the participant view steady during calibration.
6. Once calibration completes, the robot starts mirroring the remote
   participant: head pose, gaze, and facial expressions.

A live preview window shows the captured region with face landmarks
overlaid, plus the current state (`TRACKING`, `EXPRESSION_RECOGNITION`,
`SPEECH_SYNC`), iris offset, and VOR status. Press `q` in that window to
quit.

### Webcam mode

```bash
python main.py
```

Same pipeline, but driven by the local webcam instead of a screen region.

### Training

The runtime expects two trained checkpoints under `models/`. Both can be
regenerated from this repository:

```bash
# 1. Collect paired (face-landmark, routine) data, then train the expression GCN
python collect_data.py
python train_haru_inverse.py            # writes models/expression_mapping.pth

# 2. Train the VOR PG-LSTM on the ODE simulator (no real data needed)
python vor_pinn_train.py --mode train   # writes the VOR checkpoint
```

## Authors

- Fanggeng Xiong — Ocean University of China — 17685745810@163.com

## Citation

This work accompanies a paper currently under review. A full citation will be
added upon publication; for now, please cite this repository.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE)
file for details.

## Acknowledgments

The 113-point facial landmark subset and contour topology used by this
project (see `simplified_landmark_plus.py`) were originally introduced by
Hu et al. for the Emo robot platform. If you build on this project
academically, please also cite the original paper:

```bibtex
@article{hu2024human,
  title   = {Human-robot facial coexpression},
  author  = {Hu, Yuhang and Chen, Boyuan and Lin, Jiong and Wang, Yunzhe
             and Wang, Yingke and Mehlman, Cameron and Lipson, Hod},
  journal = {Science Robotics},
  volume  = {9},
  number  = {88},
  pages   = {eadi4724},
  year    = {2024},
  doi     = {10.1126/scirobotics.adi4724}
}
```

Original code release: <https://doi.org/10.5061/dryad.gxd2547t7>
