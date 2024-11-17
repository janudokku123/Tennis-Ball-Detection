# import streamlit as st

# # Title
# st.title("Tennis Game Tracking")

# # Layout with columns
# col1, col2 = st.columns([3, 1])

# # Initialize session state for preview control
# if "preview" not in st.session_state:
#     st.session_state.preview = False

# # Main Video Area
# with col1:
#     st.write("### Video")
    
#     # Drag and Drop Area
#     st.write("### Upload Video File")
#     uploaded_file = st.file_uploader("Drag and drop or browse files", type=["mp4", "mov", "avi"], key="video_upload")

#     # Display video only if file is uploaded and preview is toggled on
#     if uploaded_file is not None and st.session_state.preview:
#         st.video(uploaded_file)  # Display the video preview

# # Sidebar-like controls
# with col2:
#     st.write(" ")
#     select_file = st.button("Select Input File")
    
#     # Button to toggle preview
#     preview_video = st.button("Preview Video")

#     # Toggle session state preview based on button click
#     if preview_video and uploaded_file is not None:
#         st.session_state.preview = True
#     elif not preview_video:
#         st.session_state.preview = False

#     st.write(" ")
#     progress_bar = st.progress(0.2)  # Display initial progress as 20%
#     st.write(" ")
#     process_video = st.button("Process Video")
#     show_output = st.button("Show Output")
#     download_output = st.button("Download Output")

# # Extra note: Make sure to re-run the app after any code changes
# # Press 'R' or click 'Always rerun' in Streamlit to see changes live.


# import streamlit as st
# import time  # For simulating dynamic progress updates

# # Title
# st.title("Tennis Game Tracking")

# # Layout with columns
# col1, col2 = st.columns([3, 1])

# # Initialize session state for preview control
# if "preview" not in st.session_state:
#     st.session_state.preview = False

# # Main Video Area
# with col1:
#     st.write("### Video")
    
#     # Drag and Drop Area
#     st.write("### Upload Video File")
#     uploaded_file = st.file_uploader("Drag and drop or browse files", type=["mp4", "mov", "avi"], key="video_upload")

#     # Display video only if file is uploaded and preview is toggled on
#     if uploaded_file is not None and st.session_state.preview:
#         st.video(uploaded_file)  # Display the video preview

# # Sidebar-like controls
# with col2:
#     st.write(" ")
#     select_file = st.button("Select Input File")
    
#     # Button to toggle preview
#     preview_video = st.button("Preview Video")

#     # Toggle session state preview based on button click
#     if preview_video and uploaded_file is not None:
#         st.session_state.preview = True
#     elif not preview_video:
#         st.session_state.preview = False

#     st.write(" ")
    
#     # Set initial progress to 0%
#     progress_bar = st.progress(0)  # Start with 0% progress
#     st.write(" ")

#     process_video = st.button("Process Video")
#     show_output = st.button("Show Output")
#     download_output = st.button("Download Output")

#     # Simulating dynamic progress update during video processing
#     if process_video and uploaded_file is not None:
#         # Simulating video processing and updating progress
#         for i in range(1, 101):  # Progress updates from 1% to 100%
#             time.sleep(0.1)  # Simulate processing time
#             progress_bar.progress(i)  # Update progress bar

   

import streamlit as st
import cv2
import tempfile
import os
import requests
from roboflow import Roboflow
import time

# Your Roboflow API details
ROBOFLOW_API_KEY = "47QmI6P6A92eipzqfdLd"  # Replace with your actual API key
ROBOFLOW_MODEL_ID = "tennis_ball_detection-0u8i1"  # Replace with your actual model ID
ROBOFLOW_VERSION = 1  # Replace with the model version

# Initialize Roboflow API
roboflow = Roboflow(api_key=ROBOFLOW_API_KEY)
model = roboflow.workspace("projects-r9lz7").project("tennis_ball_detection-0u8i1").version(ROBOFLOW_VERSION)

def detect_ball_with_roboflow(input_frame):
    """
    Detects the tennis ball in the given video frame using the Roboflow API.
    """
    # Save the frame to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        temp_img_path = temp_img.name
        cv2.imwrite(temp_img_path, input_frame)
    
    # Send frame to Roboflow API for object detection
    with open(temp_img_path, "rb") as img_file:
        response = requests.post(
            f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_VERSION}",
            params={"api_key": ROBOFLOW_API_KEY},
            files={"file": img_file},
            data={"name": "test"}
        )
    
    # Parse response
    response_json = response.json()
    predictions = response_json.get("predictions", [])
    
    # Draw bounding boxes around detected balls
    for pred in predictions:
        x = int(pred["x"] - pred["width"] / 2)
        y = int(pred["y"] - pred["height"] / 2)
        width = int(pred["width"])
        height = int(pred["height"])
        confidence = pred["confidence"]
        
        # Draw rectangle and label with confidence
        cv2.rectangle(input_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(input_frame, f"{confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Clean up temporary image file
    os.remove(temp_img_path)
    return input_frame

def process_video_with_ball_detection(input_video_path):
    """
    Processes the video by detecting tennis balls in each frame using Roboflow API.
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open the video file.")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object to save processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_video_path = temp_output.name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Process each frame with Roboflow
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect tennis ball in the current frame using Roboflow
        processed_frame = detect_ball_with_roboflow(frame)
        
        # Write the processed frame into the output video file
        out.write(processed_frame)
        
        # Update progress bar
        frame_counter += 1
        progress = (frame_counter / total_frames) * 100
        st.session_state.progress = progress
    
    # Release video resources
    cap.release()
    out.release()
    
    return output_video_path

# Streamlit Interface
st.title("Tennis Game Tracking with Roboflow")

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploaded_code" not in st.session_state:
    st.session_state.uploaded_code = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "progress" not in st.session_state:
    st.session_state.progress = 0

col1, col2 = st.columns([3, 1])

with col1:
    st.write("### Video Preview")
    video_placeholder = st.empty()

with col2:
    st.write("### File Selection")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], key="file_upload")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.write(f"**Selected File:** {uploaded_file.name}")
        video_placeholder.video(uploaded_file)
    
    st.write("### Upload Your Custom Code File")
    uploaded_code = st.file_uploader("Choose a processing script or configuration file", type=["py", "txt", "yml"], key="code_upload")
    
    if uploaded_code is not None:
        st.session_state.uploaded_code = uploaded_code
        st.write(f"**Selected Code File:** {uploaded_code.name}")

    if st.button("Process Video") and st.session_state.uploaded_file is not None and st.session_state.uploaded_code is not None:
        st.write("Processing video...")

        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(st.session_state.uploaded_file.read())
            temp_input_path = temp_input.name
        
        # Save the uploaded code to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_code:
            temp_code.write(st.session_state.uploaded_code.read())
            temp_code_path = temp_code.name
        
        # Assuming the code file will modify how we process the video (this is where you can integrate custom logic)
        # For now, we're proceeding with the basic ball detection as an example
        processed_video_path = process_video_with_ball_detection(temp_input_path)
        
        if processed_video_path:
            st.session_state.processed_video_path = processed_video_path
            st.write("Video processing complete!")
            video_placeholder.video(processed_video_path)
        else:
            st.error("Failed to process the video.")
    
    # Show processing percentage
    st.progress(st.session_state.progress / 100)  # Normalize to 0-1 range
    
    if st.session_state.processed_video_path:
        with open(st.session_state.processed_video_path, "rb") as file:
            st.download_button(
                label="Download Output",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )


