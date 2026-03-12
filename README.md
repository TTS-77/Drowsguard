# 👁️ DrowsGuard — Real-Time Drowsiness Detection System

A real-time driver drowsiness detection system built with Python, MediaPipe, and Flask. Detects eye closure using a neural network blendshape model and triggers an audio alarm after 2 seconds of sustained eye closure.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?style=flat-square&logo=opencv)

---

## 🎯 What It Does

- Captures webcam feed and runs **MediaPipe Face Landmarker** on every frame
- Reads `eyeBlinkLeft` and `eyeBlinkRight` **blendshape scores** (0 = open, 1 = closed)
- Applies **median smoothing** over 4 frames + **majority vote** over 6 frames to eliminate flicker
- If both eyes confirmed closed for **2+ seconds** → plays alarm audio once
- Eyes open again → alarm stops immediately, resets for next closure
- Streams annotated video to browser via **MJPEG** and pushes detection state via **JSON polling**
- Full HTML/CSS dashboard with live blink gauges, vote window, session stats, alert log

---

## 🧠 What I Learned

- **MediaPipe Tasks API** — using the newer `mediapipe.tasks.python.vision` API (blendshapes, not the old `solutions` API)
- **Real-time computer vision** — processing webcam frames at 30fps with temporal smoothing
- **Flask backend architecture** — MJPEG streaming, JSON state endpoints, threading
- **Frontend–backend integration** — HTML dashboard consuming a Python server with no JavaScript ML libraries
- **Temporal signal processing** — median filter + majority vote to make detection robust against single-frame noise
- **Audio control with pygame** — play-once alarm that interrupts cleanly when eyes reopen

---

## 🗂️ Project Structure

```
drowsguard/
├── app.py              # Flask backend — detection engine + routes
├── index.html          # Frontend dashboard
├── alarm.mp3           # Alarm audio (you provide this)
├── face_landmarker.task # Auto-downloaded on first run (~1MB)
└── README.md
```

---

## ⚙️ How It Works

```
Webcam → OpenCV → MediaPipe Blendshapes
                        ↓
              eyeBlinkLeft + eyeBlinkRight scores
                        ↓
              Median smooth (4 frames)
                        ↓
              Majority vote (6 frames, need 4/6)
                        ↓
              Eyes closed for 2s? → Alarm!
                        ↓
              Flask streams frame (MJPEG) + state (JSON)
                        ↓
              HTML Dashboard updates in real time
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A webcam
- An `alarm.mp3` file (any audio, placed in the project folder)

### Install dependencies

```bash
pip install flask opencv-python mediapipe numpy pygame
```

### Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser and click **Start Monitoring**.

The MediaPipe model (~1MB) downloads automatically on first run.

---

## 🔧 Configuration

Edit the constants at the top of `app.py`:

| Setting | Default | Description |
|---|---|---|
| `BLINK_THRESHOLD` | `0.35` | Blendshape score above which eye = closed |
| `ALARM_TRIGGER_SECS` | `2.0` | Seconds of closed eyes before alarm |
| `SMOOTH_WINDOW` | `4` | Median filter window size |
| `VOTE_WINDOW` | `6` | Majority vote window size |
| `VOTE_MIN` | `4` | Minimum votes to confirm closure |
| `CAMERA_INDEX` | `0` | Webcam index (0 = default) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection | MediaPipe Face Landmarker (blendshapes) |
| Computer Vision | OpenCV |
| Backend | Python + Flask |
| Video Stream | MJPEG over HTTP |
| Audio | pygame |
| Frontend | Vanilla HTML / CSS / JS |

---

## 📌 Notes

- This project requires a **physical webcam** and runs locally — it cannot be fully deployed to a cloud server (no camera access on remote servers)
- For a deployed demo, the frontend runs in standalone mode using the browser's MediaPipe WebAssembly build
- Tested on Python 3.10, mediapipe 0.10.x

---

## 🔊 Custom Alarm Audio

This project uses a custom `alarm.mp3` — in my version I used a funny audio clip to wake the user up (works way better than a boring beep, trust me).

You can use **any audio file you want**:
- A funny sound effect
- Someone yelling "WAKE UP!"
- A loud siren
- Your own voice recording
- Literally anything

Just rename your file to `alarm.mp3` and drop it in the project folder. The file is excluded from git (`.gitignore`) so everyone who clones this repo brings their own audio. Only requirement: it must be an **MP3 named `alarm.mp3`**.

---

## 📄 License

MIT