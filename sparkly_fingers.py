#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Hand Gesture Recognition with Visual Effects
# Modified version with custom visual enhancements

import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import random
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def parse_configuration():
    """
    Parse command line arguments for application configuration.
    Returns configuration parameters for camera and hand tracking.
    """
    config_parser = argparse.ArgumentParser(description="Hand gesture recognition with visual effects")

    # Camera settings
    config_parser.add_argument("--device", type=int, default=0, help="Camera device index")
    config_parser.add_argument("--width", help='Camera capture width', type=int, default=960)
    config_parser.add_argument("--height", help='Camera capture height', type=int, default=540)

    # MediaPipe hand tracking settings
    config_parser.add_argument('--use_static_image_mode', action='store_true', help="Use static image mode")
    config_parser.add_argument("--min_detection_confidence",
                        help='Minimum confidence for hand detection',
                        type=float,
                        default=0.7)
    config_parser.add_argument("--min_tracking_confidence",
                        help='Minimum confidence for hand tracking',
                        type=float,
                        default=0.5)

    configuration = config_parser.parse_args()

    return configuration


def main():
    # Configuration setup #############################################################
    config = parse_configuration()

    # Camera parameters
    camera_device = config.device
    camera_width = config.width
    camera_height = config.height

    # Hand tracking parameters
    static_image_mode = config.use_static_image_mode
    min_detection_confidence = config.min_detection_confidence
    min_tracking_confidence = config.min_tracking_confidence

    # Display settings
    show_bounding_rect = True

    # Camera initialization ###############################################################
    video_capture = cv.VideoCapture(camera_device)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)

    # Initialize hand tracking model #############################################
    mediapipe_hands = mp.solutions.hands
    hand_tracker = mediapipe_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,  # Allow detection of both hands
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Initialize gesture classifiers #############################################
    gesture_classifier = KeyPointClassifier()
    motion_classifier = PointHistoryClassifier()

    # Load gesture label data ###################################################
    # Hand pose labels (static gestures)
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as label_file:
        gesture_labels_reader = csv.reader(label_file)
        gesture_labels = [
            row[0] for row in gesture_labels_reader
        ]

    # Motion gesture labels (dynamic gestures)
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as label_file:
        motion_labels_reader = csv.reader(label_file)
        motion_labels = [
            row[0] for row in motion_labels_reader
        ]

    # Performance monitoring ################################################
    fps_calculator = CvFpsCalc(buffer_len=15)  # Increased buffer for smoother FPS display

    # Tracking data structures ###############################################
    # Finger position history for trail effect - separate tracking for each hand
    tracking_history_length = 16
    finger_position_history = {
        "Left": deque(maxlen=tracking_history_length),
        "Right": deque(maxlen=tracking_history_length)
    }

    # Initialize the history queues with placeholder values
    for _ in range(tracking_history_length):
        finger_position_history["Left"].append([0, 0])
        finger_position_history["Right"].append([0, 0])

    # Gesture detection history for smoothing - separate for each hand
    gesture_history = {
        "Left": deque(maxlen=tracking_history_length),
        "Right": deque(maxlen=tracking_history_length)
    }

    # Initialize gesture history with zeros
    for _ in range(tracking_history_length):
        gesture_history["Left"].append(0)
        gesture_history["Right"].append(0)

    # Application state #######################################################
    app_mode = 0  # 0: Normal mode, 1: Keypoint logging, 2: Motion logging

    # Main processing loop #####################################################
    while True:
        # Calculate and display current FPS
        current_fps = fps_calculator.get()

        # Handle keyboard input ################################################
        pressed_key = cv.waitKey(10)
        if pressed_key == 27:  # ESC key to exit
            break
        input_number, app_mode = process_keyboard_input(pressed_key, app_mode)

        # Capture frame from camera #############################################
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Flip horizontally for a more natural interaction (mirror effect)
        frame = cv.flip(frame, 1)

        # Create a copy for visualization
        display_image = copy.deepcopy(frame)

        # Hand detection process #################################################
        # Convert to RGB for MediaPipe processing
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Disable image writeable to improve performance
        rgb_frame.flags.writeable = False

        # Process the frame with MediaPipe Hands
        detection_results = hand_tracker.process(rgb_frame)

        # Re-enable image writeable
        rgb_frame.flags.writeable = True

        # Process detected hands #################################################
        if detection_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(detection_results.multi_hand_landmarks,
                                                 detection_results.multi_handedness):
                # Get hand side (Left or Right)
                hand_side = handedness.classification[0].label

                # Calculate hand bounding box
                hand_rect = calc_bounding_rect(display_image, hand_landmarks)

                # Extract landmark coordinates
                landmark_list = calc_landmark_list(display_image, hand_landmarks)

                # Preprocess landmarks for classification
                normalized_landmark_list = pre_process_landmark(landmark_list)

                # Get the appropriate history for this hand
                current_hand_history = finger_position_history[hand_side]

                # Process motion history for this specific hand
                normalized_motion_history = pre_process_point_history(
                    display_image, current_hand_history)

                # Log data for training (if in logging mode)
                logging_csv(input_number, app_mode, normalized_landmark_list,
                           normalized_motion_history)

                # Classify hand pose (static gesture)
                hand_gesture_id = gesture_classifier(normalized_landmark_list)

                # Track index finger position for sparkle effect
                if hand_gesture_id == 2:  # Point gesture detected
                    # Store position of index finger tip (landmark 8)
                    finger_position_history[hand_side].append(landmark_list[8])
                else:
                    # No pointing gesture, add placeholder
                    finger_position_history[hand_side].append([0, 0])

                # Classify finger motion (dynamic gesture)
                motion_gesture_id = 0
                history_data_points = len(normalized_motion_history)

                # Only classify when we have enough history data
                if history_data_points == (tracking_history_length * 2):
                    motion_gesture_id = motion_classifier(normalized_motion_history)

                # Update gesture history for smoothing
                gesture_history[hand_side].append(motion_gesture_id)
                most_common_gesture = Counter(gesture_history[hand_side]).most_common()

                # Visualization section
                # Draw hand outline and bounding box
                display_image = draw_bounding_rect(show_bounding_rect, display_image, hand_rect)
                display_image = draw_landmarks(display_image, landmark_list)

                # Display gesture information
                display_image = draw_info_text(
                    display_image,
                    hand_rect,
                    handedness,
                    gesture_labels[hand_gesture_id],
                    motion_labels[most_common_gesture[0][0]],
                )
        else:
            # No hands detected, add placeholders to maintain history
            finger_position_history["Left"].append([0, 0])
            finger_position_history["Right"].append([0, 0])

        # Add visual effects and information overlay
        display_image = visualize_finger_trail(display_image, finger_position_history)
        display_image = draw_info(display_image, current_fps, app_mode, input_number)

        # Display the result #########################################################
        cv.imshow('Hand Gesture Recognition with Effects', display_image)

    # Clean up resources
    video_capture.release()
    cv.destroyAllWindows()
    print("Application closed successfully")


def process_keyboard_input(key_code, current_mode):
    """
    Process keyboard input to change application mode and handle number inputs.

    Args:
        key_code: The key code from cv.waitKey()
        current_mode: The current application mode

    Returns:
        tuple: (input_number, new_mode)
            input_number: Number pressed (0-9) or -1 if no number was pressed
            new_mode: Updated application mode
    """
    input_number = -1

    # Handle number keys (0-9)
    if 48 <= key_code <= 57:
        input_number = key_code - 48

    # Handle mode switching keys
    if key_code == 110:  # 'n' key - Normal mode
        current_mode = 0
    elif key_code == 107:  # 'k' key - Keypoint logging mode
        current_mode = 1
    elif key_code == 104:  # 'h' key - History logging mode
        current_mode = 2

    return input_number, current_mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_points):
    """
    Draws hand skeleton and key points on the image.

    Args:
        image: The frame to draw on
        landmark_points: List of landmark coordinates

    Returns:
        Image with hand landmarks visualized
    """
    if len(landmark_points) > 0:
        # Define colors for the hand visualization
        outline_color = (100, 50, 150)  # Darker purple for outline
        line_color = (180, 105, 255)    # Lighter purple for main lines

        # Thumb connections
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
                outline_color, 5)  # Thicker outline
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
                line_color, 2)     # Thinner colored line
        cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                line_color, 2)

        # Index finger connections
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                line_color, 2)

        # Middle finger connections
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                line_color, 2)

        # Ring finger connections
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                line_color, 2)

        # Little finger connections
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                line_color, 2)

        # Palm connections
        cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                line_color, 2)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                outline_color, 5)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                line_color, 2)

    # Key Points
    for index, landmark in enumerate(landmark_points):
        if index == 0:  # Wrist point 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Wrist point 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Little finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Little finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Little finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Little finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(show_rect, image, rect_coords):
    """
    Draws a bounding rectangle around the detected hand.

    Args:
        show_rect: Boolean flag to determine if rectangle should be drawn
        image: The frame to draw on
        rect_coords: Coordinates of the rectangle [x1, y1, x2, y2]

    Returns:
        Image with bounding rectangle
    """
    if show_rect:
        x1, y1, x2, y2 = rect_coords

        # Draw rectangle with purple color to match theme
        # Outer rectangle with thicker line
        cv.rectangle(image, (x1, y1), (x2, y2), (180, 105, 255), 2)

        # Inner rectangle with white color for contrast
        cv.rectangle(image, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), 1)

    return image


def draw_info_text(image, hand_rect, handedness, hand_gesture_text, motion_gesture_text):
    """
    Draws gesture recognition information on the image.

    Args:
        image: The frame to draw on
        hand_rect: Bounding rectangle of the hand
        handedness: Hand laterality information (left/right)
        hand_gesture_text: Recognized static hand gesture
        motion_gesture_text: Recognized dynamic motion gesture

    Returns:
        Image with gesture information overlay
    """
    # Create semi-transparent overlay for hand info
    x1, y1, x2, y2 = hand_rect
    overlay = image.copy()

    # Background for hand gesture label
    cv.rectangle(overlay, (x1, y1-30), (x2, y1), (50, 50, 50), -1)
    cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Combine hand laterality and gesture text
    hand_side = handedness.classification[0].label[0:]
    if hand_gesture_text != "":
        display_text = f"{hand_side}: {hand_gesture_text}"
    else:
        display_text = hand_side

    # Draw hand gesture text with purple theme
    cv.putText(image, display_text, (x1 + 5, y1 - 8),
              cv.FONT_HERSHEY_SIMPLEX, 0.65, (180, 105, 255), 2, cv.LINE_AA)
    cv.putText(image, display_text, (x1 + 5, y1 - 8),
              cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv.LINE_AA)

    # Display motion gesture if available
    if motion_gesture_text != "":
        # Create background for motion gesture text
        motion_overlay = image.copy()
        cv.rectangle(motion_overlay, (10, 45), (350, 75), (50, 50, 50), -1)
        cv.addWeighted(motion_overlay, 0.7, image, 0.3, 0, image)

        # Draw motion gesture text
        motion_text = f"Motion: {motion_gesture_text}"
        cv.putText(image, motion_text, (20, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 0.75, (180, 105, 255), 2, cv.LINE_AA)
        cv.putText(image, motion_text, (20, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)

    return image


def create_glow_effect(image, position, radius=35, intensity=0.6, color=(180, 105, 255)):
    """
    Creates an enhanced glowing halo effect around the sparkle.

    Args:
        image: The frame or image to draw on.
        position: Tuple (x, y) - the center of the glow.
        radius: The maximum radius of the glow.
        intensity: Intensity of the glow (0.0 to 1.0).
        color: Color of the glow in (B, G, R) format - default is purple.
    """
    # Create a copy for overlay blending
    glow_layer = image.copy()
    center_x, center_y = position

    # Create multi-layer glow with decreasing opacity
    for radius_step in range(radius, 0, -4):  
        # Calculate opacity based on distance from center
        opacity = intensity * (radius_step / radius) * 0.8

        # Draw filled circle for the glow
        cv.circle(glow_layer, (center_x, center_y), radius_step, color, thickness=-1)

        # Blend the glow with the original image
        image = cv.addWeighted(glow_layer, opacity, image, 1 - opacity, 0)

    return image


def render_sparkle_effect(image, position, size=18, primary_color=(180, 105, 255), secondary_color=(255, 255, 255)):
    """
    Renders an enhanced sparkle effect at the specified position.
    Creates a more dynamic and visually appealing sparkle with multiple colors.

    Args:
        image: The frame or image to draw on.
        position: (x, y) position of the center of the sparkle.
        size: Base size of the sparkle.
        primary_color: Main color of the sparkle in (B, G, R) format - default is purple.
        secondary_color: Highlight color in (B, G, R) format - default is white.
    """
    center_x, center_y = position

    # Create primary rays (longer)
    for i in range(6):  # 6 main rays
        # Calculate angle with even spacing
        angle = i * (2 * math.pi / 6)

        # Add slight randomness to length
        ray_length = size * random.uniform(1.2, 1.8)

        # Calculate end point
        end_x = int(center_x + ray_length * math.cos(angle))
        end_y = int(center_y + ray_length * math.sin(angle))

        # Draw the primary ray with thickness 2
        cv.line(image, (center_x, center_y), (end_x, end_y), primary_color, thickness=2)

    # Create secondary rays (shorter, different color)
    for i in range(8):  # 8 secondary rays
        # Calculate angle with offset from primary rays
        angle = (i * (2 * math.pi / 8)) + (math.pi / 8)

        # Shorter rays with randomness
        ray_length = size * random.uniform(0.5, 0.9)

        # Calculate end point
        end_x = int(center_x + ray_length * math.cos(angle))
        end_y = int(center_y + ray_length * math.sin(angle))

        # Draw the secondary ray
        cv.line(image, (center_x, center_y), (end_x, end_y), secondary_color, thickness=1)


def visualize_finger_trail(image, position_histories):
    """
    Creates visual effects along the paths traced by fingers from both hands.
    Renders sparkles and optional glow effects at each tracked position.

    Args:
        image: The frame or image to draw on.
        position_histories: Dictionary with hand labels as keys and lists of (x,y) positions as values.

    Returns:
        The image with visual effects added for all detected hands.
    """
    # Process each hand's history
    for hand_side, position_history in position_histories.items():
        # Different colors for different hands
        if hand_side == "Left":
            primary_color = (180, 105, 255)  # Purple for left hand
        else:
            primary_color = (255, 105, 180)  # Pink for right hand

        # Process each point in this hand's history
        for index, point in enumerate(position_history):
            # Skip placeholder points (when no finger was detected)
            if point[0] != 0 and point[1] != 0:
                # Add sparkle effect at this position
                render_sparkle_effect(
                    image, 
                    (point[0], point[1]), 
                    size=20, 
                    primary_color=primary_color,
                    secondary_color=(255, 255, 255)  # White highlights
                )

                # Uncomment to add glow effect (can be performance intensive)
                # create_glow_effect(
                #     image, 
                #     (point[0], point[1]), 
                #     radius=35, 
                #     intensity=0.6, 
                #     color=primary_color
                # )

    return image


def draw_info(image, fps, mode, number):
    """
    Draws application information overlay on the image.

    Args:
        image: The frame to draw on
        fps: Current frames per second
        mode: Application mode
        number: Input number from keyboard

    Returns:
        Image with information overlay
    """
    # Create semi-transparent overlay for text background
    info_overlay = image.copy()
    cv.rectangle(info_overlay, (5, 5), (250, 40), (50, 50, 50), -1)
    cv.addWeighted(info_overlay, 0.6, image, 0.4, 0, image)

    # Draw FPS with purple color scheme
    cv.putText(image, f"FPS: {fps}", (15, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (180, 105, 255), 3, cv.LINE_AA)  # Shadow
    cv.putText(image, f"FPS: {fps}", (15, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 1, cv.LINE_AA)  # Text

    # Mode information
    mode_names = ['Normal Mode', 'Keypoint Training', 'Motion Training']

    if 0 <= mode <= 2:
        # Create another overlay for mode info if in special mode
        if mode > 0:
            mode_overlay = image.copy()
            cv.rectangle(mode_overlay, (5, 60), (280, 120), (50, 50, 50), -1)
            cv.addWeighted(mode_overlay, 0.7, image, 0.3, 0, image)

            # Display current mode
            cv.putText(image, f"MODE: {mode_names[mode]}", (15, 85),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 105, 255), 2, cv.LINE_AA)

            # Display selected number if applicable
            if 0 <= number <= 9:
                cv.putText(image, f"TRAINING CLASS: {number}", (15, 115),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
