# import streamlit as st
# import torch
# import cv2
# import tempfile
# import numpy as np
# import pathlib
# from pathlib import Path
# import requests

# # Load the custom YOLOv5 model (best.pt should be in the same folder as app.py)
# MODEL_PATH = "best.pt"  # Path to your trained YOLOv5 model weights
# #model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, source='local')
# #model = torch.hub.load('C:/Users/Admin/Downloads/tennis_videos/yolov5', 'custom', path=MODEL_PATH, source='local')
# model = torch.hub.load('C:/Users/Admin/Downloads/tennis_videos/yolov5', 'custom', path=MODEL_PATH, source='local', force_reload=True)



# # Streamlit app UI
# st.title('Tennis Ball Detection App')
# st.write('Upload a tennis video to detect the tennis ball in real-time.')

# # File uploader for video input
# uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# if uploaded_video is not None:
#     # Save the uploaded video to a temporary file
#     temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video_path = str(temp_video.name)
#     temp_video.write(uploaded_video.read())
#     temp_video.close()

#     # Open video capture
#     cap = cv2.VideoCapture(temp_video.name)
    
#     # Create a progress bar
#     progress_bar = st.progress(0)
#     stframe = st.empty()  # Placeholder to show video frames

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     frame_idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Perform detection on the current frame
#         results = model(frame)  # Pass the frame to the model for detection
        
#         # Render the results (draw boxes around detected objects)
#         frame = np.squeeze(results.render())  # Get the frame with drawn bounding boxes

#         # Convert BGR (OpenCV format) to RGB (Streamlit format)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the processed frame
#         stframe.image(frame, channels='RGB', use_column_width=True)
        
#         # Update the progress bar
#         frame_idx += 1
#         progress = frame_idx / total_frames
#         progress_bar.progress(progress)

#     cap.release()
#     st.success('Video processing complete!')

#     # Option to download the processed video
#     processed_video_path = "/path/to/save/processed_video.mp4"  # Update this path to where you want to save
#     out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

#     cap = cv2.VideoCapture(temp_video.name)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)
#         frame = np.squeeze(results.render())
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Write the frame to the output video
#         out.write(frame)
    
#     cap.release()
#     out.release()

#     # Create a download link for the processed video
#     with open(processed_video_path, "rb") as f:
#         st.download_button(
#             label="Download Processed Video",
#             data=f,
#             file_name="processed_video.mp4",
#             mime="video/mp4"
#         )

# else:
#     st.warning('Please upload a video file to begin detection.')


# import torch
# import streamlit as st
# import cv2
# import tempfile
# import numpy as np
# import pathlib
# import os
# from pathlib import Path
# import requests
# import shutil

# # Directly import YOLOv5 from the cloned repository
# import sys
# sys.path.append('C:/Users/Admin/Downloads/tennis_videos/yolov5')  # Add yolov5 repo to path

# from models.experimental import attempt_load  # Import model loading function from YOLOv5 repo

# # Define the correct path to your custom model
# MODEL_PATH = 'C:/Users/Admin/Downloads/tennis_videos/best.pt'  # Path to your best.pt file

# # Load the custom YOLOv5 model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use CUDA if available
# model = attempt_load(MODEL_PATH, map_location=device)  # Load the model to the device

# # Streamlit app UI
# st.title('Tennis Player Detection App')
# st.write('Upload a tennis video to detect players in real-time.')

# # File uploader for video input
# uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# if uploaded_video is not None:
#     # Save the uploaded video to a temporary file
#     temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video_path = str(temp_video.name)
#     temp_video.write(uploaded_video.read())
#     temp_video.close()

#     # Open video capture
#     cap = cv2.VideoCapture(temp_video.name)
    
#     # Get the total number of frames for progress bar
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Initialize progress bar
#     progress_bar = st.progress(0)

#     # Create a temporary file to save the processed video
#     processed_video_path = 'processed_video.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (640, 480))

#     stframe = st.empty()

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection
#         results = model(frame)  # Perform detection using the YOLOv5 model
#         frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

#         # Convert BGR to RGB for display
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Write the processed frame to the output video
#         out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#         # Update the progress bar
#         frame_count += 1
#         progress_bar.progress(frame_count / total_frames)

#         # Display the frame in Streamlit
#         stframe.image(frame, channels='RGB', use_column_width=True)

#     cap.release()
#     out.release()

#     # Show a success message after processing is complete
#     st.success('Video processing complete!')

#     # Provide a download link for the processed video
#     with open(processed_video_path, 'rb') as f:
#         st.download_button(
#             label="Download Processed Video",
#             data=f,
#             file_name="processed_video.mp4",
#             mime="video/mp4"
#         )

#     # Clean up temporary files
#     os.remove(temp_video_path)
#     os.remove(processed_video_path)

# import streamlit as st
# import torch
# from pathlib import Path
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# from io import BytesIO

# # Set device (GPU or CPU)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load YOLOv5 model using torch.hub
# @st.cache_resource  # Cache the model loading
# def load_model():
#     model_path = 'C:/Users/Admin/Downloads/tennis_videos/best.pt'  # Replace with your model's path
#     model = torch.hub.load('C:/Users/Admin/Downloads/tennis_videos/yolov5', 'custom', path=model_path, source='local')
#     model.eval()  # Set to evaluation mode
#     return model

# # Load the model
# model = load_model()

# # Streamlit UI for the app
# st.title("Tennis Ball Detection")

# st.write("""
#     This is a simple web app that detects tennis balls in videos using a custom-trained YOLOv5 model.
#     Upload a video file, and the app will process it frame by frame.
# """)

# # File uploader widget for video upload
# video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

# # Progress bar for processing
# progress_bar = st.progress(0)
# status_text = st.empty()

# if video_file is not None:
#     # Create a temporary file for the uploaded video
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(video_file.read())
#         video_path = tmp_file.name

#     # Open the uploaded video
#     cap = cv2.VideoCapture(video_path)

#     # Get video details
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Prepare the output video file
#     output_path = "output_video.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

#     frame_idx = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform inference with YOLOv5
#         img = Image.fromarray(frame)
#         results = model(img)
        
#         # Update progress bar
#         frame_idx += 1
#         progress_bar.progress(frame_idx / frame_count)
#         status_text.text(f"Processing frame {frame_idx} of {frame_count}")

#         # Draw bounding boxes (if ball is detected)
#         for *xyxy, conf, cls in results.xyxy[0]:  # xyxy: (xmin, ymin, xmax, ymax)
#             if cls == 0:  # Ball class is usually '0' in COCO
#                 label = f"Ball {conf:.2f}"
#                 cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Write the frame to the output video
#         out.write(frame)

#     # Release video capture and writer
#     cap.release()
#     out.release()

#     # Provide download option for processed video
#     with open(output_path, "rb") as file:
#         video_bytes = file.read()
    
#     st.download_button(label="Download Processed Video", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")

#     # Display the output video as an embedded video player
#     st.video(output_path)

#     # Delete the temporary video file
#     os.remove(video_path)

#Working
# import streamlit as st
# import torch
# import cv2
# import tempfile
# import numpy as np
# import pathlib
# from pathlib import Path

# # Fix for Windows path compatibility
# pathlib.PosixPath = pathlib.WindowsPath

# # Update the paths to your local directories
# repo_path = r'C:\Users\Admin\Downloads\tennis_videos\yolov5'  # Replace with your YOLOv5 repo path
# model_path = r'C:\Users\Admin\Downloads\tennis_videos\best.pt'  # Replace with your actual .pt file path

# # Load the custom YOLOv5 model from local path
# model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# # Streamlit app UI
# st.title('Tennis Player Detection App')
# st.write('Upload a tennis video to detect players in real-time.')

# # File uploader for video input
# uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# if uploaded_video is not None:
#     # Save the uploaded video to a temporary file
#     temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video_path = str(temp_video.name)
#     temp_video.write(uploaded_video.read())
#     temp_video.close()

#     # Open video capture
#     cap = cv2.VideoCapture(temp_video.name)
#     stframe = st.empty()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection
#         results = model(frame)
#         frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

#         # Convert BGR to RGB for display
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame in Streamlit
#         stframe.image(frame, channels='RGB', use_column_width=True)

#     cap.release()
#     st.success('Video processing complete!')

# st.write("Ensure 'best.pt' is in the correct path or provide the correct path in model_path.")

import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
from pathlib import Path

# Fix for Windows path compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Update the paths to your local directories
repo_path = r'C:\Users\Admin\Downloads\tennis_videos\yolov5'  # Replace with your YOLOv5 repo path
model_path = r'C:\Users\Admin\Downloads\tennis_videos\best.pt'  # Replace with your actual .pt file path

# Load the custom YOLOv5 model from local path
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Streamlit app UI
st.title('Tennis Player Detection App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = str(temp_video.name)
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video.name)
    
    # Prepare for saving the processed video
    output_video_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Progress bar
    progress_bar = st.progress(0)
    stframe = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video processing loop
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to output video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Update progress bar
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

    # Release resources
    cap.release()
    out.release()

    # Success message and video download link
    st.success('Video processing complete!')
    
    # Provide download link for the processed video
    with open(output_video_path, "rb") as file:
        st.download_button("Download Processed Video", file, file_name="processed_video.mp4")

st.write("Ensure 'best.pt' is in the correct path or provide the correct path in model_path.")
