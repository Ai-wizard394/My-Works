import pyaudio
import webrtcvad
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # Chunk size in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Chunk size in samples
WINDOW_DURATION = 5000  # Duration in milliseconds to monitor dual voices
WINDOW_SIZE = int(RATE * WINDOW_DURATION / 1000 / CHUNK_SIZE)  # Number of chunks in the window
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressive mode for voice activity detection

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables for face and eye tracking
last_detected_faces = []
last_detected_eyes = []

# Global variables for GUI
root = tk.Tk()
root.title("Dual Voice and Face Detection")

# Load and configure background image
bg_image = Image.open("C:\\ML_\\Remote-Exam-Proctoring-2.png")
bg_image = bg_image.resize((800, 600), Image.LANCZOS)  # Use Image.LANCZOS for antialiasing
bg_image_tk = ImageTk.PhotoImage(bg_image)

video_label = tk.Label(root, image=bg_image_tk)
video_label.pack(padx=10, pady=10)

video_stream = None
stream = None
is_detecting = False

def start_detection():
    global stream, video_stream, is_detecting
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    video_stream = cv2.VideoCapture(0)  # Adjust index if using a different camera
    video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    is_detecting = True
    
    # Start a new thread for detecting voices and updating GUI
    detection_thread = threading.Thread(target=detect_dual_voices)
    detection_thread.start()

def stop_detection():
    global stream, video_stream, is_detecting
    is_detecting = False
    if stream:
        stream.stop_stream()
        stream.close()
    if video_stream:
        video_stream.release()
    cv2.destroyAllWindows()

def detect_faces(gray_frame):
    # Detect faces with different scales and conditions for better detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    gray_frame = cv2.equalizeHist(gray_frame)
    
    return gray_frame

def detect_dual_voices():
    global is_detecting, video_label
    
    window = []  # Buffer to store recent chunks
    voice_count = 0  # Counter for active voices
    reset_time = 5  # Time in seconds to reset back to normal
    last_speech_time = 0
    cheating_detected = False
    last_cheating_report_time = 0

    while is_detecting:
        try:
            # Read audio data
            data = stream.read(CHUNK_SIZE)
            chunk = np.frombuffer(data, dtype=np.int16)
            
            # Check for voice activity
            is_speech = vad.is_speech(chunk.tobytes(), RATE)
            
            if is_speech:
                window.append(chunk)
                voice_count += 1  # Increment voice count for active voices

                # Maintain window size
                if len(window) > WINDOW_SIZE:
                    window.pop(0)  # Remove oldest chunk
                
                # Check for dual or multiple voices in the window
                if len(window) == WINDOW_SIZE:
                    voices_detected = sum(vad.is_speech(chunk.tobytes(), RATE) for chunk in window)
                    
                    if voices_detected >= 2:  # Adjust threshold for multiple voices detection
                        last_speech_time = time.time()  # Update last speech time
                        if voice_count >= 3 and not cheating_detected:  # Adjust threshold as needed
                            cheating_detected = True
                            last_cheating_report_time = time.time()
                            print("Cheating Detected! Multiple voices detected.")
                    
                    else:
                        voice_count = 0  # Reset counter if no multiple voices detected
                        if cheating_detected and time.time() - last_cheating_report_time > reset_time:
                            cheating_detected = False
                            print("Everything is good.")  # Indicate everything is good
            
            else:
                voice_count = 0  # Reset counter if no speech detected
                if cheating_detected and time.time() - last_cheating_report_time > reset_time:
                    cheating_detected = False
                    print("Everything is good.")  # Indicate everything is good
            
            # Read video frame
            ret, frame = video_stream.read()
            if not ret:
                print("Error reading video frame.")
                break
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Preprocess frame for better eye detection
            gray_frame = preprocess_frame(frame)

            # Detect faces
            faces = detect_faces(gray_frame)

            # Track faces and eyes across frames
            global last_detected_faces, last_detected_eyes
            last_detected_eyes = []
            
            # Report multiple faces detected
            if len(faces) > 1:
                print(f"Multiple faces detected: {len(faces)}")
                last_detected_faces = faces
                cheating_detected = True  # Report cheating if multiple faces detected
            elif len(faces) == 1:
                # Track position if single face detected
                last_detected_faces = faces
            else:
                # Reset if no faces detected
                last_detected_faces = []
            
            for (x, y, w, h) in last_detected_faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (ex, ey, ew, eh) in eyes:
                    ex_abs = ex + x
                    ey_abs = ey + y
                    last_detected_eyes.append((ex_abs, ey_abs, ew, eh))

                    # Example: Check if eyes are in unnatural position (example only, needs calibration)
                    if ew < w * 0.15 or eh < h * 0.15:
                        cheating_detected = True
                        print("Cheating Detected! Unnatural eye movement.")
                    
                    # Determine eye position relative to face center
                    eye_center_x = ex_abs + ew // 2
                    face_center_x = x + w // 2
                    eye_position = "Center"
                    
                    if eye_center_x < face_center_x - 20:
                        eye_position = "Left"
                    elif eye_center_x > face_center_x + 20:
                        eye_position = "Right"
                    
                    # Draw eye position on the frame
                    cv2.putText(frame, f"Eye Position: {eye_position}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw rectangles around detected faces and eyes
            for (x, y, w, h) in last_detected_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (ex, ey, ew, eh) in last_detected_eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Overlay message on video frame based on cheating detection
            if cheating_detected:
                cv2.putText(frame, "Cheating Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Everything is good.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert the frame to RGB format and then to ImageTk format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update the video_label with the new image
            video_label.configure(image=img_tk)
            video_label.image = img_tk
            
            root.update_idletasks()  # Update the GUI window
            
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    if stream:
        stream.stop_stream()
        stream.close()
    is_detecting = False

def main():
    start_btn = ttk.Button(root, text="Start Detection", command=start_detection)
    start_btn.pack(pady=10)
    
    stop_btn = ttk.Button(root, text="Stop Detection", command=stop_detection)
    stop_btn.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
