AUTOMATIC DETECTION OF FRAUDULENT IMAGE AND DOCUMENTS USING DEEP LEARNING

A web-based application that uses a hybrid approach to detect digital forgeries. It leverages a powerful pre-trained deep learning model to identify deepfakes in images and videos, and uses Error Level Analysis (ELA) to spot potential manipulations in documents.

Key Features:
Dual Analysis Modes: Users can choose between two distinct analysis types:
•	Image DeepFake Detection: Utilizes an AI model to analyze images and faces for signs of deepfake manipulation.
•	Document Forgery Detection: Employs Error Level Analysis (ELA) to identify edited or copy-pasted sections within document images (e.g., JPEGs).
•	Simple Web Interface: An easy-to-use interface built with Flask allows users to upload an image and receive an instant analysis.
•	Quantitative Scoring: Provides a confidence score for deepfakes and a forgery score for documents to help quantify the analysis.
•	Visual Feedback: Generates a visual heatmap to highlight areas of potential manipulation.

How It Works:
This project combines two different techniques to create a more robust detection system.
1. DeepFake Detection (AI Model)
For standard images and faces, the application uses the tf_efficientnet_b7_ns model, a state-of-the-art image classification model made available through the timm library. This model was pre-trained on millions of images and is highly effective at identifying the subtle digital artifacts and inconsistencies that are characteristic of deepfakes. The model outputs a probability score indicating the likelihood that the image is a deepfake.

2. Document Forgery Detection (ELA)
For documents, the application switches to Error Level Analysis (ELA). This technique works by detecting discrepancies in the JPEG compression levels across an image. When a section of an image is edited, copied, or inserted, it is often saved at a different compression level than the rest of the image. ELA highlights these mismatched areas, making them visible in a heatmap. The application calculates a "forgery score" based on the intensity of these discrepancies.

Installation and Setup:
Follow these steps to run the application on your local machine.
Prerequisites:-
•	Python 3.8 or newer.
•	pip (Python package installer).

Step-by-Step Guide:-
•	Clone the Repository
        git clone (https://github.com/Neha-Akter-Alisha/Deepfake-Backend.git)
        cd your-repository-name

•	Create and Activate a Virtual Environment
        It is highly recommended to use a virtual environment to manage project dependencies.

•	Create the environment
        python -m venv venv

•	Activate the environment (Windows)
         .\venv\Scripts\Activate

•	Install Dependencies
        All required packages are listed in the requirements.txt file.
        pip install -r requirements.txt
(Note: This step may take several minutes as torch and torchvision are large libraries.)

•	Run the Application
         Once the installation is complete, you can start the Flask server.
         python app.py
•	Access the Application
         Open your web browser and navigate to:
         https://www.google.com/search?q=http://127.0.0.1:5000

How to Use:-
•	Open the web application in your browser.
•	Select the analysis type: "Face / Image DeepFake" or "Document Forgery".
•	Click "Choose File" and select an image from your computer.
•	Click the "Analyze Image" button.
•	The results page will display the analysis, the score, and the original and processed images.

Technologies Used:
•	Backend: Flask
•	Deep Learning: PyTorch, Timm (PyTorch Image Models)
•	Image & Document Processing:
•	Pillow (PIL): Used for core image manipulation and implementing the Error Level Analysis (ELA) for document forgery.
•	OpenCV: Used for general image handling tasks.
•	NumPy: Used for numerical operations to calculate the forgery score.
•	Frontend: HTML, CSS

License
This project is licensed under the MIT License. See the LICENSE file for more details.
