from flask import Flask, request, send_file
import os
import cv2
import numpy as np
import mediapipe as mp
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to detect lunge
def detect_lunge(left_knee_angle, right_knee_angle, leg_angle):
    if 80 < left_knee_angle < 110 and 80 < right_knee_angle < 110:
        if 80 < leg_angle < 110:
            return "Lunge Detected: Correct", (0, 255, 0)  # Green
        else:
            return "Lunge Detected but Legs Not Open Properly", (0, 255, 255)  # Yellow
    else:
        return "Incorrect Lunge or No Lunge", (0, 0, 255)  # Red

# Function to detect sit-up
def detect_sit_up(left_knee_angle, right_knee_angle):
    if left_knee_angle < 70 and right_knee_angle < 70:
        return "Sit up Detected: Correct", (0, 255, 0)  # Green
    else:
        return "Incorrect Sit Up or No Sit Up", (0, 0, 255)  # Red

def process_video(pose_id, video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'output_{uuid.uuid4()}.mp4'  # Unique filename
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            pose_text = "No Pose Detected"
            color = (255, 255, 255)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height]

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                if pose_id == 1:
                    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
                    leg_angle = calculate_angle(left_knee, mid_hip, right_knee)
                    pose_text, color = detect_lunge(left_knee_angle, right_knee_angle, leg_angle)
                elif pose_id == 2:
                    pose_text, color = detect_sit_up(left_knee_angle, right_knee_angle)

                cv2.putText(image, f"Left Knee Angle: {int(left_knee_angle)}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image, f"Right Knee Angle: {int(right_knee_angle)}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image, pose_text, (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(image)

        return output_path
    finally:
        cap.release()
        out.release()

@app.route('/process-video', methods=['POST'])
def process_video_route():
    try:
        if 'video' not in request.files:
            return "No video file provided", 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return "No selected video file", 400

        pose_id = int(request.form.get('pose_id', 1))
        
        # Save uploaded video temporarily
        temp_path = f'temp_{uuid.uuid4()}.mp4'
        video_file.save(temp_path)
        
        # Process video
        output_path = process_video(pose_id, temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Send processed video back with cleanup callback
        response = send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='processed_video.mp4'
        )
        
        # Delete the output file after sending
        response.call_on_close(lambda: os.remove(output_path))
        return response
        
    except Exception as e:
        return str(e), 500
    finally:
        # Ensure temp file is deleted even if an error occurs
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)