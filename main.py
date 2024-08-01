import os
import numpy as np
import cv2
import torch
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline
import openai
from datetime import datetime

openai.api_key = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')

# Define the classes
classes = ["Defective", "Non defective"]

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.25  # confidence threshold

# Load the ResNet50 model without the top layer for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load the trained logistic regression model
model_filename = 'logistic_regression_model.joblib'
clf = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load a pre-trained transformer model for text summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def preprocess_frame_for_yolo(frame):
    img = Image.fromarray(frame)
    return img

def detect_anomalies(frame):
    img = preprocess_frame_for_yolo(frame)
    results = yolo_model(img)
    return results

def draw_bounding_box(frame, bbox, confidence):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{confidence:.2f}"
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return frame

def preprocess_frame_for_classification(frame_path):
    img = Image.open(frame_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_frame(frame_path):
    img_array = preprocess_frame_for_classification(frame_path)
    feature = base_model.predict(img_array).flatten().reshape(1, -1)
    prediction = clf.predict(feature)
    confidence = clf.predict_proba(feature)
    return prediction[0], confidence[0]

def process_video_and_save_screenshots(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # List to store bounding box coordinates
    bbox_coords = []
    screenshot_folder = 'vid_photos'
    os.makedirs(screenshot_folder, exist_ok=True)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_anomalies(frame)

        for detection in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            frame = draw_bounding_box(frame, (int(x1), int(y1), int(x2), int(y2)), conf)
            # Save the coordinates and confidence
            bbox_coords.append({'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 
                                'confidence': conf, 
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]})
            # Save screenshot
            screenshot_path = os.path.join(screenshot_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(screenshot_path, frame)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the bounding box coordinates to a file in the screenshot folder
    output_coords_path = os.path.join(screenshot_folder, 'bounding_box_coords.txt')
    with open(output_coords_path, 'w') as f:
        for bbox in bbox_coords:
            f.write(f"Frame {bbox['frame']}, Confidence: {bbox['confidence']:.2f}, BBox: {bbox['bbox']}\n")

def classify_saved_screenshots(screenshot_folder):
    defect_count = 0
    total_frames = 0

    for filename in os.listdir(screenshot_folder):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(screenshot_folder, filename)
            prediction, confidence = classify_frame(frame_path)
            total_frames += 1
            if classes[prediction] == "Defective":
                defect_count += 1

            # Debug: print classification results
            print(f"{filename}: {classes[prediction]} ({confidence[prediction]:.2f})")

            # Optionally, save the frame with annotation
            if classes[prediction] == "Defective":
                annotated_frame = cv2.imread(frame_path)
                text = f"{classes[prediction]}: {confidence[prediction]:.2f}"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(screenshot_folder, f"defective_{filename}"), annotated_frame)

    defect_percentage = (defect_count / total_frames) * 100 if total_frames > 0 else 0
    st.write(f"Total Frames Analyzed: {total_frames}")
    st.write(f"Defective Frames: {defect_count}")
    st.write(f"Defective Percentage: {defect_percentage:.2f}%")
        
    # Generate a textual summary using the pre-trained transformer model
    summary_text = generate_summary_openai(total_frames, defect_count, defect_percentage)
    st.write(summary_text)
    
    # Display the image with track defects
    st.image('/Users/janekkorczynski/Desktop/BizHack/Screenshot 2024-07-25 at 10.28.21.png', caption='Track that needs maintenance')

    # Generate rerouting suggestions
    rerouting_text = generate_rerouting_suggestions()
    st.write(rerouting_text)

def generate_summary_openai(total_frames, defect_count, defect_percentage):
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    We have analyzed {total_frames} frames of railway tracks in a video. Out of these, {defect_count} frames were found to be defective, which represents {defect_percentage:.2f}% of the total frames analyzed.

    To: Team Leader, Railway Maintenance
    From: Automated Analysis System
    Date: {current_date}

    1. **Safety Risks Associated with Detected Defects**
       The analysis has revealed a significant rate of defects—{defect_count} frames out of {total_frames} analyzed ({defect_percentage:.2f}%). The potential safety risks tied to these defects include:
       - **Derailment**: Structural weaknesses can lead to train derailments, endangering passengers and freight.
       - **Accidents at Level Crossings**: Defective tracks can affect signaling systems, potentially leading to accidents at crossings.
       - **Passenger Safety**: Jarring movements caused by uneven tracks may lead to injuries.
       - **Infrastructure Damage**: Defects can contribute to further deterioration of surrounding infrastructure, leading to increased safety hazards.
       Immediate attention to these defects is imperative to ensure the safety of all railway operations.

    2. **Importance of Timely Maintenance**
       Timely maintenance is critical in preventing the escalation of these defects into more serious issues. The impact of unaddressed defects includes:
       - **Increased Operational Delays**: Defective tracks may necessitate slower train speeds or reroutes, causing delays.
       - **Higher Repair Costs**: Neglecting minor defects often results in more significant damages, leading to elevated repair costs.
       - **Operational Challenges**: Frequent inspections and unexpected repairs can disrupt scheduled operations, impacting service.

    3. **Recommendations for Maintenance**
       - **Immediate Actions**: Conduct an urgent inspection of the defective sections identified in the analysis. Prioritize the defects based on their severity and potential impact on operations.
       - **Maintenance Plan**: Develop a detailed maintenance plan to address the identified defects. This should include scheduling repair work, allocating necessary resources, and ensuring that all safety protocols are followed.
       - **Team Coordination**: Mobilize and coordinate your maintenance team efficiently. Ensure clear communication of tasks and safety procedures to all team members.
       - **Long-term Strategy**: Implement a long-term strategy for ongoing monitoring and maintenance of the railway tracks. Use advanced detection systems to regularly check for defects and prevent similar issues in the future.
       - **Documentation**: Keep thorough records of all inspections, repairs, and maintenance activities. This documentation will be valuable for future reference and continuous improvement.

    By following these recommendations, we can enhance the safety and reliability of our railway operations.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed and actionable summaries for maintenance team leaders."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message['content'].strip()

def generate_rerouting_suggestions():
    prompt = """
    Given the closed route (Red line) on the Bangalore Metro Map, please provide detailed rerouting suggestions for trains traveling from Kempegowda Stn, Majestic to Indiranagar.
    When the red line is visible on the closes way to that station. Change the route and include alternative routes, transfers, and any other relevant information to ensure a smooth journey.
    Come up with an alterative solutions of travel. Do not describe all of the train lines, just the lines that are important to make a journey possible. 
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed and insightful summaries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0]['message']['content'].strip()


# Streamlit UI
st.title('Railway Defect Detection')
st.write('Upload a video file to analyze for defects.')

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create the uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    input_video_path = os.path.join("uploads", uploaded_file.name)
    output_video_path = "output_video.mp4" 

    # Save the uploaded file
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing the video...")
    process_video_and_save_screenshots(input_video_path, output_video_path)
    
    st.write("Classifying the saved screenshots...")
    classify_saved_screenshots('vid_photos')
    
    # st.write("Analysis complete. The results are displayed above.")
