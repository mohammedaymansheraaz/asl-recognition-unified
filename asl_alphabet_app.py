#!/usr/bin/env python3
"""
ASL Alphabet Recognition App
============================
Static ASL letter recognition (A-Z) using TensorFlow Lite + MediaPipe HandLandmarker
"""

from pathlib import Path
import csv
import copy
import itertools

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Alphabet Model Paths (Static Letters - A-Z)
ALPHA_MODEL_DIR = BASE_DIR / "models" / "alphabet"
ALPHA_TFLITE_PATH = ALPHA_MODEL_DIR / "keypoint_classifier.tflite"
ALPHA_LABELS_PATH = ALPHA_MODEL_DIR / "keypoint_classifier_label.csv"
ALPHA_HAND_TASK_PATH = ALPHA_MODEL_DIR / "hand_landmarker.task"


# =============================================================================
# ALPHABET MODEL CONSTANTS (Static Letters)
# =============================================================================
WINDOW_NAME = "ASL Alphabet Recognition (A-Z)"


# =============================================================================
# UTILITIES
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


def draw_ui(image, fps):
    """Draw UI for alphabet model."""
    cv2.putText(
        image, "ASL Alphabet (A-Z)", (10, 25),
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
    cv2.putText(
        image, "ESC=quit  |  l=toggle landmarks", (10, image.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
    )
    return image


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class ASLAlphabetApp:
    def __init__(self):
        self.show_landmarks = True
        self.cap = None
        self.timestamp = 0

        # Alphabet model components
        self.classifier = None
        self.labels = []
        self.landmarker = None
        self.fps_calc = None

    def load_model(self):
        """Load alphabet model and dependencies."""
        print("Loading Alphabet Model (26 Static Letters A-Z)...")
        self.classifier = KeyPointClassifier(str(ALPHA_TFLITE_PATH))

        with open(ALPHA_LABELS_PATH, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader]

        self.landmarker = create_hand_landmarker()

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

        self.fps_calc = CvFpsCalc(buffer_len=10)

        print(f"Loaded alphabet model: {ALPHA_TFLITE_PATH}")
        print(f"Labels: {self.labels}")

    def init_camera(self, device=0, width=960, height=540):
        """Initialize webcam."""
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def run_frame(self, frame):
        """Process one frame."""
        debug_image = copy.deepcopy(frame)
        fps = self.fps_calc.get()

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)

        self.timestamp += 1
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp)

        if detection_result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                detection_result.hand_landmarks, detection_result.handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classification
                hand_sign_id = self.classifier(pre_processed_landmark_list)
                hand_sign_text = self.labels[hand_sign_id] if hand_sign_id < len(self.labels) else "?"

                # Drawing
                if self.show_landmarks:
                    debug_image = draw_alpha_bounding_rect(True, debug_image, brect)
                    debug_image = draw_alpha_landmarks(debug_image, landmark_list)
                debug_image = draw_alpha_info_text(debug_image, brect, handedness, hand_sign_text)

        debug_image = draw_ui(debug_image, fps)
        return debug_image

    def run(self):
        """Main application loop."""
        print("\n" + "=" * 50)
        print("ASL ALPHABET RECOGNITION (A-Z)")
        print("=" * 50)
        print("Model loaded successfully!")
        print("\nControls:")
        print("  ESC  - Quit")
        print("  l    - Toggle landmark visualization")
        print("=" * 50)

        self.init_camera()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = self.run_frame(frame)

                cv2.imshow(WINDOW_NAME, processed_frame)

                key = cv2.waitKey(10) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord("l"):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="ASL Alphabet Recognition App")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Camera width")
    parser.add_argument("--height", type=int, default=540, help="Camera height")
    args = parser.parse_args()

    app = ASLAlphabetApp()
    app.load_model()
    app.run()


if __name__ == "__main__":
    main()