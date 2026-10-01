#!/usr/bin/env python3
"""
Merged ASL Recognition App
==========================
Combines two ASL recognition models:
1. VOCABULARY MODEL (Dynamic Signs) - 24 ASL words using LSTM + MediaPipe Holistic
2. ALPHABET MODEL (Static Letters) - 26 ASL letters (A-Z) using TFLite + MediaPipe HandLandmarker

Press 'w' to switch between models.
"""

from pathlib import Path
import urllib.request
import csv
import copy
import itertools
import argparse

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Vocabulary Model Paths (Dynamic Signs - 24 words)
VOCAB_MODEL_DIR = BASE_DIR / "models" / "vocabulary"
VOCAB_MODEL_PATH = VOCAB_MODEL_DIR / "best.keras"
VOCAB_ACTIONS_PATH = VOCAB_MODEL_DIR / "actions.npy"
VOCAB_MEAN_PATH = VOCAB_MODEL_DIR / "mean.npy"
VOCAB_STD_PATH = VOCAB_MODEL_DIR / "std.npy"
VOCAB_HOLISTIC_TASK_PATH = VOCAB_MODEL_DIR / "holistic_landmarker.task"
VOCAB_HOLISTIC_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)

# Alphabet Model Paths (Static Letters - A-Z)
ALPHA_MODEL_DIR = BASE_DIR / "models" / "alphabet"
ALPHA_TFLITE_PATH = ALPHA_MODEL_DIR / "keypoint_classifier.tflite"
ALPHA_LABELS_PATH = ALPHA_MODEL_DIR / "keypoint_classifier_label.csv"
ALPHA_HAND_TASK_PATH = ALPHA_MODEL_DIR / "hand_landmarker.task"


# =============================================================================
# VOCABULARY MODEL CONSTANTS (Dynamic Signs)
# =============================================================================
VOCAB_WINDOW_NAME = "ASL Recognition - Vocabulary Mode (24 Words)"
VOCAB_THRESHOLD = 0.60
VOCAB_STABLE_FRAME_COUNT = 8
VOCAB_MAX_HISTORY_WORDS = 8

# Selected face landmarks for vocabulary model
SELECTED_FACE_LANDMARKS = [
    # mouth / lips
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88,
    91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
    308, 310, 311, 312, 317, 318, 321, 324, 375, 402, 405,

    # eyes / eyebrow-ish region
    33, 46, 52, 53, 55, 65, 66, 70, 105, 107,
    133, 145, 159, 160, 161, 263, 276, 282, 283,
    285, 295, 296, 300, 334, 336, 362, 374, 386, 387, 388,

    # nose / center face
    1, 2, 4, 5, 6, 98, 327,
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]

FACE_CONNECTIONS = [
    # outer / upper mouth
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 291),

    # lower mouth
    (61, 146), (146, 91), (91, 181), (181, 17),
    (17, 405), (405, 321), (321, 375), (375, 291),

    # inner mouth
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),

    # left eye
    (33, 160), (160, 159), (159, 145), (145, 133), (133, 33),

    # right eye
    (263, 387), (387, 386), (386, 374), (374, 362), (362, 263),

    # left eyebrow-ish
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 70),
    (70, 105), (105, 107),

    # right eyebrow-ish
    (276, 283), (283, 282), (282, 295), (295, 285), (285, 300),
    (300, 334), (334, 336),

    # nose / center
    (1, 2), (2, 4), (4, 5), (5, 6), (98, 1), (327, 1),
]


# =============================================================================
# ALPHABET MODEL CONSTANTS (Static Letters)
# =============================================================================
ALPHA_WINDOW_NAME = "ASL Recognition - Alphabet Mode (A-Z)"


# =============================================================================
# SHARED UTILITIES
# =============================================================================
def download_vocab_task_model():
    """Download the holistic landmarker task model if not present."""
    VOCAB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if VOCAB_HOLISTIC_TASK_PATH.exists():
        return

    print("Downloading holistic_landmarker.task...")
    urllib.request.urlretrieve(VOCAB_HOLISTIC_TASK_URL, VOCAB_HOLISTIC_TASK_PATH)
    print("Download complete.")


def to_point(landmark, width, height):
    """Convert normalized landmark to pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)


# =============================================================================
# VOCABULARY MODEL FUNCTIONS (Dynamic Signs)
# =============================================================================
def create_holistic_landmarker():
    """Create MediaPipe Holistic Landmarker for vocabulary model."""
    download_vocab_task_model()

    base_options = python.BaseOptions(
        model_asset_path=str(VOCAB_HOLISTIC_TASK_PATH)
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    return vision.HolisticLandmarker.create_from_options(options)


def extract_vocab_keypoints(results):
    """Extract keypoints for vocabulary model (pose + hands + selected face)."""
    pose = np.zeros(33 * 3, dtype=np.float32)
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)
    face = np.zeros(len(SELECTED_FACE_LANDMARKS) * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks[:33]
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks],
            dtype=np.float32,
        ).flatten()

    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks[:21]
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in left_hand_landmarks],
            dtype=np.float32,
        ).flatten()

    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks[:21]
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in right_hand_landmarks],
            dtype=np.float32,
        ).flatten()

    if results.face_landmarks:
        selected_face_values = []
        for idx in SELECTED_FACE_LANDMARKS:
            if idx < len(results.face_landmarks):
                lm = results.face_landmarks[idx]
                selected_face_values.extend([lm.x, lm.y, lm.z])
            else:
                selected_face_values.extend([0.0, 0.0, 0.0])
        face = np.array(selected_face_values, dtype=np.float32)

    keypoints = np.concatenate([pose, lh, rh, face]).astype(np.float32)
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    return keypoints


def normalize_vocab_sequence(sequence, mean, std):
    """Normalize vocabulary sequence using training statistics."""
    sequence = np.array(sequence, dtype=np.float32)
    return (sequence - mean) / std


def draw_vocab_face_skeleton(
    image,
    landmarks,
    selected_landmarks,
    connections,
    point_color=(180, 180, 180),
    line_color=(160, 160, 160),
    point_radius=1,
    line_thickness=1,
):
    """Draw face skeleton for vocabulary model."""
    if not landmarks:
        return

    h, w = image.shape[:2]
    selected_set = set(selected_landmarks)

    for start_idx, end_idx in connections:
        if start_idx not in selected_set or end_idx not in selected_set:
            continue
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start = to_point(landmarks[start_idx], w, h)
        end = to_point(landmarks[end_idx], w, h)

        cv2.line(
            image, start, end, line_color, line_thickness, cv2.LINE_AA
        )

    for idx, lm in enumerate(landmarks):
        if idx in selected_set:
            point = to_point(lm, w, h)
            cv2.circle(
                image, point, point_radius, point_color, -1, cv2.LINE_AA
            )


def draw_vocab_skeleton(
    image,
    landmarks,
    connections,
    point_color,
    line_color,
    point_radius=2,
    line_thickness=1,
):
    """Draw skeleton for vocabulary model."""
    if not landmarks:
        return

    h, w = image.shape[:2]

    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start = to_point(landmarks[start_idx], w, h)
        end = to_point(landmarks[end_idx], w, h)

        cv2.line(
            image, start, end, line_color, line_thickness, cv2.LINE_AA
        )

    for lm in landmarks:
        point = to_point(lm, w, h)
        cv2.circle(
            image, point, point_radius, point_color, -1, cv2.LINE_AA
        )


def draw_vocab_results(image, results, show_landmarks):
    """Draw all landmarks for vocabulary model."""
    if not show_landmarks:
        return

    draw_vocab_face_skeleton(
        image=image,
        landmarks=results.face_landmarks,
        selected_landmarks=SELECTED_FACE_LANDMARKS,
        connections=FACE_CONNECTIONS,
        point_color=(180, 180, 180),
        line_color=(160, 160, 160),
        point_radius=1,
        line_thickness=1,
    )

    draw_vocab_skeleton(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        point_color=(0, 255, 255),
        line_color=(0, 180, 255),
        point_radius=2,
        line_thickness=1,
    )

    draw_vocab_skeleton(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        point_color=(0, 255, 0),
        line_color=(0, 180, 0),
        point_radius=3,
        line_thickness=1,
    )

    draw_vocab_skeleton(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        point_color=(255, 80, 80),
        line_color=(255, 80, 80),
        point_radius=3,
        line_thickness=1,
    )


def draw_vocab_ui(frame, live_text, sentence, mode_name):
    """Draw UI for vocabulary model."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)

    # Mode indicator
    cv2.putText(
        frame,
        f"Mode: {mode_name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Live: {live_text}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    history_text = "History: " + " ".join(sentence[-VOCAB_MAX_HISTORY_WORDS:])
    if len(history_text) > 85:
        history_text = history_text[:82] + "..."

    cv2.putText(
        frame,
        history_text,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# ALPHABET MODEL FUNCTIONS (Static Letters)
# =============================================================================
class KeyPointClassifier:
    """TensorFlow Lite classifier for alphabet model."""
    def __init__(self, model_path, num_threads=1):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index


def create_hand_landmarker():
    """Create MediaPipe Hand Landmarker for alphabet model."""
    base_options = python.BaseOptions(model_asset_path=str(ALPHA_HAND_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def calc_bounding_rect(image, landmarks):
    """Calculate bounding rectangle for hand landmarks."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """Convert landmarks to pixel coordinates list."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """Pre-process landmarks: relative coordinates + normalization."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))

    return temp_landmark_list


def draw_alpha_landmarks(image, landmark_point):
    """Draw hand landmarks for alphabet model."""
    if len(landmark_point) == 0:
        return image

    connections = [
        # Thumb
        (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Little finger
        (17, 18), (18, 19), (19, 20),
        # Palm
        (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmark_point) and end_idx < len(landmark_point):
            cv2.line(image, tuple(landmark_point[start_idx]), tuple(landmark_point[end_idx]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[start_idx]), tuple(landmark_point[end_idx]), (255, 255, 255), 2)

    # Draw key points
    for index, landmark in enumerate(landmark_point):
        if index in [0, 1, 2, 5, 9, 13, 17]:  # wrist and finger bases
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        elif index in [4, 8, 12, 16, 20]:  # fingertips
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        else:  # other joints
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image


def draw_alpha_bounding_rect(use_brect, image, brect):
    """Draw bounding rectangle for alphabet model."""
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_alpha_info_text(image, brect, handedness, hand_sign_text):
    """Draw info text for alphabet model."""
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    # handedness is a list of Handedness objects
    if handedness and len(handedness) > 0:
        info_text = handedness[0].category_name[0:]
    else:
        info_text = "Unknown"
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text

    cv2.putText(
        image, info_text, (brect[0] + 5, brect[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return image


def draw_alpha_ui(image, fps, mode_name):
    """Draw UI for alphabet model."""
    cv2.putText(
        image, f"Mode: {mode_name}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        image, "FPS:" + str(fps), (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        image, "FPS:" + str(fps), (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )
    return image


# =============================================================================
# MAIN MERGED APPLICATION
# =============================================================================
class MergedASLApp:
    def __init__(self):
        self.mode = "vocab"  # "vocab" or "alpha"
        self.show_landmarks = True
        self.cap = None
        self.timestamp = 0

        # Vocabulary model components
        self.vocab_model = None
        self.vocab_actions = None
        self.vocab_mean = None
        self.vocab_std = None
        self.vocab_sequence_length = None
        self.vocab_feature_size = None
        self.vocab_landmarker = None
        self.vocab_sequence = []
        self.vocab_sentence = []
        self.vocab_last_prediction = ""
        self.vocab_stable_count = 0

        # Alphabet model components
        self.alpha_classifier = None
        self.alpha_labels = []
        self.alpha_landmarker = None
        self.alpha_fps_calc = None

    def load_vocab_model(self):
        """Load vocabulary model and dependencies."""
        print("Loading Vocabulary Model (24 Dynamic Signs)...")
        self.vocab_model = load_model(VOCAB_MODEL_PATH)
        self.vocab_actions = np.load(VOCAB_ACTIONS_PATH, allow_pickle=True)
        self.vocab_mean = np.load(VOCAB_MEAN_PATH)
        self.vocab_std = np.load(VOCAB_STD_PATH)

        self.vocab_sequence_length = self.vocab_model.input_shape[1]
        self.vocab_feature_size = self.vocab_model.input_shape[2]

        self.vocab_mean = np.squeeze(self.vocab_mean)
        self.vocab_std = np.squeeze(self.vocab_std)
        self.vocab_std = np.where(self.vocab_std == 0, 1.0, self.vocab_std)

        # Verify feature size matches
        expected_face_features = len(SELECTED_FACE_LANDMARKS) * 3
        expected_feature_size = (33 * 3) + (21 * 3) + (21 * 3) + expected_face_features
        if self.vocab_feature_size != expected_feature_size:
            raise ValueError(
                f"Feature size mismatch. Model expects {self.vocab_feature_size}, "
                f"but code builds {expected_feature_size}."
            )

        self.vocab_landmarker = create_holistic_landmarker()

        print(f"Loaded vocabulary model: {VOCAB_MODEL_PATH}")
        print(f"Actions: {self.vocab_actions}")
        print(f"Model input shape: {self.vocab_model.input_shape}")
        print(f"Selected face landmarks: {len(SELECTED_FACE_LANDMARKS)}")

    def load_alpha_model(self):
        """Load alphabet model and dependencies."""
        print("Loading Alphabet Model (26 Static Letters A-Z)...")
        self.alpha_classifier = KeyPointClassifier(str(ALPHA_TFLITE_PATH))

        with open(ALPHA_LABELS_PATH, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            self.alpha_labels = [row[0] for row in reader]

        self.alpha_landmarker = create_hand_landmarker()

        # Simple FPS calculator
        class CvFpsCalc:
            def __init__(self, buffer_len=10):
                self._start_tick = cv2.getTickCount()
                self._freq = 1000.0 / cv2.getTickFrequency()
                self._frame_times = []
                self._buffer_len = buffer_len

            def get(self):
                current_tick = cv2.getTickCount()
                elapsed_time = (current_tick - self._start_tick) * self._freq
                self._start_tick = current_tick
                self._frame_times.append(elapsed_time)
                if len(self._frame_times) > self._buffer_len:
                    self._frame_times.pop(0)
                if len(self._frame_times) > 0:
                    return round(1000.0 / (sum(self._frame_times) / len(self._frame_times)))
                return 0

        self.alpha_fps_calc = CvFpsCalc(buffer_len=10)

        print(f"Loaded alphabet model: {ALPHA_TFLITE_PATH}")
        print(f"Labels: {self.alpha_labels}")

    def load_models(self):
        """Load both models."""
        self.load_vocab_model()
        self.load_alpha_model()

    def init_camera(self):
        """Initialize webcam."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    def switch_mode(self):
        """Switch between vocabulary and alphabet mode."""
        if self.mode == "vocab":
            self.mode = "alpha"
            print("\n=== Switched to ALPHABET MODE (A-Z Static Letters) ===")
            print("Controls: ESC=quit, w=switch mode, l=toggle landmarks")
        else:
            self.mode = "vocab"
            print("\n=== Switched to VOCABULARY MODE (24 Dynamic Signs) ===")
            print("Controls: ESC=quit, w=switch mode, c=clear history, l=toggle landmarks")
            # Reset vocabulary sequence when switching back
            self.vocab_sequence = []
            self.vocab_sentence = []
            self.vocab_last_prediction = ""
            self.vocab_stable_count = 0

    def run_vocab_frame(self, frame):
        """Process one frame in vocabulary mode."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.vocab_landmarker.detect(mp_image)
        draw_vocab_results(frame, results, self.show_landmarks)

        keypoints = extract_vocab_keypoints(results)

        if keypoints.shape[0] != self.vocab_feature_size:
            return frame, "Feature mismatch"

        self.vocab_sequence.append(keypoints)
        self.vocab_sequence = self.vocab_sequence[-self.vocab_sequence_length:]

        live_text = "Collecting frames..."

        if len(self.vocab_sequence) == self.vocab_sequence_length:
            input_sequence = normalize_vocab_sequence(
                self.vocab_sequence, self.vocab_mean, self.vocab_std
            )
            input_data = np.expand_dims(input_sequence, axis=0).astype(np.float32)

            prediction = self.vocab_model.predict(input_data, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_action = str(self.vocab_actions[predicted_index])
            confidence = float(prediction[predicted_index])

            if confidence >= VOCAB_THRESHOLD:
                live_text = f"{predicted_action} ({confidence:.2f})"

                if predicted_action == self.vocab_last_prediction:
                    self.vocab_stable_count += 1
                else:
                    self.vocab_stable_count = 0
                    self.vocab_last_prediction = predicted_action

                if self.vocab_stable_count >= VOCAB_STABLE_FRAME_COUNT:
                    if len(self.vocab_sentence) == 0 or self.vocab_sentence[-1] != predicted_action:
                        self.vocab_sentence.append(predicted_action)

                    self.vocab_sentence = self.vocab_sentence[-VOCAB_MAX_HISTORY_WORDS:]
            else:
                live_text = f"Uncertain ({confidence:.2f})"

        draw_vocab_ui(frame, live_text, self.vocab_sentence, "VOCABULARY (24 Words)")
        return frame, live_text

    def run_alpha_frame(self, frame):
        """Process one frame in alphabet mode."""
        debug_image = copy.deepcopy(frame)
        fps = self.alpha_fps_calc.get()

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)

        self.timestamp += 1
        detection_result = self.alpha_landmarker.detect_for_video(mp_image, self.timestamp)

        if detection_result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                detection_result.hand_landmarks, detection_result.handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classification
                hand_sign_id = self.alpha_classifier(pre_processed_landmark_list)
                hand_sign_text = self.alpha_labels[hand_sign_id] if hand_sign_id < len(self.alpha_labels) else "?"

                # Drawing
                debug_image = draw_alpha_bounding_rect(True, debug_image, brect)
                debug_image = draw_alpha_landmarks(debug_image, landmark_list)
                debug_image = draw_alpha_info_text(debug_image, brect, handedness, hand_sign_text)

        debug_image = draw_alpha_ui(debug_image, fps, "ALPHABET (A-Z Letters)")
        return debug_image, f"FPS: {fps}"

    def run(self):
        """Main application loop."""
        print("\n" + "=" * 60)
        print("MERGED ASL RECOGNITION APP")
        print("=" * 60)
        print("Models loaded successfully!")
        print("\n=== Starting in VOCABULARY MODE (24 Dynamic Signs) ===")
        print("Controls:")
        print("  ESC  - Quit")
        print("  w    - Switch between Vocabulary/Alphabet mode")
        print("  c    - Clear history (Vocab mode only)")
        print("  l    - Toggle landmark visualization")
        print("=" * 60)

        self.init_camera()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                # Process frame based on current mode
                if self.mode == "vocab":
                    processed_frame, _ = self.run_vocab_frame(frame)
                    window_name = VOCAB_WINDOW_NAME
                else:
                    processed_frame, _ = self.run_alpha_frame(frame)
                    window_name = ALPHA_WINDOW_NAME

                cv2.imshow(window_name, processed_frame)

                # Handle key presses
                key = cv2.waitKey(10) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord("w"):  # Switch mode
                    self.switch_mode()
                    # Close old window, will create new one on next imshow
                    cv2.destroyAllWindows()
                elif key == ord("c") and self.mode == "vocab":  # Clear history
                    self.vocab_sentence = []
                    self.vocab_stable_count = 0
                    self.vocab_last_prediction = ""
                elif key == ord("l"):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.vocab_landmarker:
            self.vocab_landmarker.close()
        if self.alpha_landmarker:
            self.alpha_landmarker.close()
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Merged ASL Recognition App")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Camera width")
    parser.add_argument("--height", type=int, default=540, help="Camera height")
    args = parser.parse_args()

    app = MergedASLApp()
    app.load_models()
    app.run()


if __name__ == "__main__":
    main()