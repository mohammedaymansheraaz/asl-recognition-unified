# ASL Recognition Unified

Real-time American Sign Language recognition with **two models in one app**:

1. **Vocabulary Model** - 24 dynamic ASL words (LSTM + MediaPipe Holistic)
2. **Alphabet Model** - 26 static ASL letters A-Z (TFLite + MediaPipe HandLandmarker)

Press **`w`** to switch between models.

---

## Models

| Model | Type | Recognizes | Input |
|-------|------|------------|-------|
| **Vocabulary** | Keras LSTM | 24 words | 40-frame sequences (pose + hands + face) |
| **Alphabet** | TFLite | 26 letters (A-Z) | Single frame (hands only) |

**Vocabulary words:** bath, bye, dad, drink, food, happy, hello, hungry, mad, milk, mom, no, please, sad, see, sick, sleep, thankyou, thirsty, wait, water, where, yes, yourself

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run merged app (both models, press `w` to switch)
```bash
python asl_merged_app.py --device 0
```

### Run alphabet-only app (lighter, A-Z only)
```bash
python asl_alphabet_app.py --device 0
```

### Options
```bash
--device 0       # Camera index (0=built-in, 1/2=mobile webcam)
--width 960      # Camera width
--height 540     # Camera height
```

---

## Controls

| Key | Merged App | Alphabet App |
|-----|------------|--------------|
| `ESC` | Quit | Quit |
| `w` | **Switch Vocabulary ↔ Alphabet** | — |
| `c` | Clear history (Vocab mode) | — |
| `l` | Toggle landmarks | Toggle landmarks |

---

## Project Structure

```
asl-recognition-unified/
├── asl_merged_app.py          # Main app (both models, 'w' to switch)
├── asl_alphabet_app.py        # Alphabet-only app (A-Z)
├── requirements.txt
├── models/
│   ├── vocabulary/            # 24-word dynamic signs model
│   │   ├── best.keras         # Trained LSTM model
│   │   ├── actions.npy        # 24 sign labels
│   │   ├── mean.npy           # Normalization mean
│   │   ├── std.npy            # Normalization std
│   │   └── holistic_landmarker.task  # MediaPipe Holistic
│   └── alphabet/              # 26-letter static signs model
│       ├── keypoint_classifier.tflite    # TFLite model
│       ├── keypoint_classifier_label.csv # A-Z labels
│       └── hand_landmarker.task          # MediaPipe Hands
```

---

## Mobile Webcam (DroidCam / Iriun / Phone Link)

1. Start mobile webcam app on phone
2. Start Windows companion app
3. Find camera index:
   ```bash
   python -c "import cv2; [print(f'Cam {i}: OK') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
   ```
4. Run with `--device <index>` (usually 1 or 2)

---

## Model Details

### Vocabulary Model (Dynamic Signs)
- **Architecture:** 2-layer LSTM (128 → 64 units) + Dense layers
- **Input:** 40 frames × 444 features (33 pose + 21 left hand + 21 right hand + 73 face landmarks × 3)
- **Training:** Adam, lr=0.0005, categorical crossentropy
- **Accuracy:** 71% on 24 classes

### Alphabet Model (Static Letters)
- **Architecture:** TFLite classifier on hand landmarks
- **Input:** 21 hand landmarks × 2 (x,y) = 42 features (normalized)
- **Classes:** 26 (A-Z)

---

## License

MIT License - See original repositories for model licenses.