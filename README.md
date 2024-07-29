# Railway Defect Detection and Rerouting Suggestions Application - BizHack-Hackathon-Project

## Overview

This project is a sketch and idea developed during a Hackathon organized by Infosys, where I had the pleasure to participate. The application analyzes videos of railway tracks to detect defects and provides suggestions for maintenance and rerouting in case of track closures.

## Prerequisites

1. **Python**: Ensure Python 3.9+ is installed on your system.
2. **Dependencies**: Install the following Python packages:
   - `numpy`
   - `opencv-python`
   - `torch`
   - `joblib`
   - `streamlit`
   - `Pillow`
   - `tensorflow`
   - `transformers`
   - `openai`
   - `networkx`
   - `matplotlib`

   You can install these packages using pip:
   ```bash
   pip install numpy opencv-python torch joblib streamlit Pillow tensorflow transformers openai networkx matplotlib
   ```

## Setting Up the Environment

1. **OpenAI API Key**: Set up the OpenAI API key as an environment variable. This key is required to access the OpenAI API to generate textual summaries. Add the following lines to your `.bashrc`, `.zshrc`, or equivalent shell configuration file:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key_here'
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key. Then, run the following command to load the new environment variable:
   ```bash
   source ~/.bashrc
   # or
   source ~/.zshrc
   ```

2. **Directory Structure**: Ensure your working directory has the following structure:
   ```
   project_root/
   ├── main.py
   ├── logistic_regression_model.joblib
   ├── bangalore_positions_short.json
   ├── closures.json
   ├── path/to/track_image.png
   └── uploads/
   ```
   - `main.py`: The main script file.
   - `logistic_regression_model.joblib`: Pre-trained logistic regression model.
   - `bangalore_positions_short.json`: JSON file containing station positions.
   - `closures.json`: JSON file containing closed connections.
   - `path/to/track_image.png`: Image showing the track that needs maintenance.
   - `uploads/`: Directory where uploaded video files will be stored.

## Running the Application

1. **Start Streamlit**: Run the Streamlit application using the following command in the terminal:
   ```bash
   streamlit run main.py
   ```

2. **Upload a Video**: Once the Streamlit application is running, open your web browser and go to `http://localhost:8501`. You will see the Railway Defect Detection application interface.
   - **Upload a Video File**: Click on "Choose a video file..." and upload your video file (formats supported: mp4, avi, mov).

3. **Video Processing**: The application processes the video to detect defects using two deep learning models:
   - **YOLOv5 (You Only Look Once)**: This model is used for real-time object detection. It analyzes each frame of the video to identify and localize defects in the railway tracks by drawing bounding boxes around the detected anomalies.
   - **ResNet50 (Residual Network)**: After YOLOv5 detects and localizes potential defects, the ResNet50 model is used for feature extraction and further classification of the detected areas to confirm if they are indeed defective.

   The processed frames are saved as images in the "vid_photos" directory, and the coordinates of the detected anomalies are saved in a text file within the same directory. The processed frames are also compiled into a new video file.

4. **Screenshot Classification**: The saved screenshots are classified to identify defective frames:
   - Each screenshot is preprocessed and passed through the ResNet50 model for feature extraction.
   - The extracted features are then classified using a pre-trained logistic regression model to determine if the frame is defective or non-defective.
   - The total number of frames analyzed and the number of defective frames are counted, and the percentage of defective frames is calculated.

5. **Generate Summary**: A detailed summary of the analysis results is generated and displayed. The summary includes:
   - The total number of frames analyzed and the number of defective frames.
   - The percentage of defective frames.
   - An assessment of potential safety risks associated with the detected defects.
   - The importance of timely maintenance.
   - Specific recommendations for addressing the defects found in the railway tracks, including immediate actions and long-term strategies.
   - Additionally, the OpenAI API key is used to leverage a Language Learning Model (LLM) to generate understandable text, making it easier for people to comprehend the technical issues and recommendations.

6. **Rerouting**: The rerouting code finds and visualizes the shortest path in a railway network, considering closed connections:
   - Station and closure information is loaded from JSON files.
   - A graph representing the railway network is constructed with nodes as stations and edges as connections.
   - Dijkstra's algorithm is used to calculate the shortest path between two stations, avoiding closed connections.
   - The railway network is visualized with highlighted paths, showing the shortest route and closed connections, ensuring efficient rerouting for maintenance and repairs.
  
**License**
This project is licensed under the MIT License. For more details, please look at the [LICENSE](LICENSE) file.

**Contributing**
If you would like to contribute to this project, please feel free to reach out. Contributions are welcome and appreciated!
