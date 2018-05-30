# IMPROVED VEHICLE COUNTING AND CLASSIFICATION ON MONOCLUAR TRAFFIC VIDEO SEQUENCES
This project is a computer vision based vehicle counting system. The system is capable of performing vehicle detetcion, tracking, counting, classification (into light medium and heavy), speed estimation and traffic flow estimation. It harnesses the power of computer vision to get deeper insights into our traffic. The focus is on development of vehicle counting techniques and their comparative analysis on datasets.

## Getting Started
Download a copy of the project onto the system using a web browser or terminal commands. Unzip the porject and you are good to go.

### Prerequisites
Python v3.5+ <br />
OpenCV - Open Source Computer Vision v3.4+  <br />
Anaconda (Create a separate environment for your project) <br />

Use the following commands to install the packages into your environment: <br />
conda env create -f environment.yaml <br />
source activate cv <br />

Or you can install everything globally. Search for step by step guides to install OpenCV. The dependencies will be installed on the way. <br />

## Files in the box
Why do I see so many files and what are their roles? <br />
Here's an overview of the files. <br />

Feature based detection: <br />
vehicle_detection_kmeans.py : Vehicle detetcion using k-means clustering on FAST features <br />
vehicle_detetcion_hclustering.py : Vehicle detetcion using hierarchical clustering on FAST features <br />
Blob based detection and tracking: <br />
single_ref_line.py : Vehicle Counting using Single Reference Line <br />
multiple_reference_lines.py : Vehicle Counting using Multiple Refernce Lines <br />
region_based.py : Region based Vehicle Counting System, includes a bonus speed estimation module as well! <br />
run.py : The final system. Multiple reference lines based counting coupled with speed estimation and traffic flow estimation.<br /> 
All the files have the module for vehicle classfification and video writing. 

## Let's run this thing

Activate your environment. Change the directory to the project Folder. Create an Input and Results folder. Place the input video in the Input folder. Run the file using python <br />
Example: <br />
python run.py <br />
python single_ref_line.py <br />
