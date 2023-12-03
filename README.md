<h1>AutoBot - Advanced Vehicle Identification System</h1>
<h2><a href="https://youtu.be/NCGko0xxi4U?feature=shared">Click here for a video demonstration of AutoBot</a></h2>

<h2>Description</h2>
<p>AutoBot leverages Computer Vision and AI to transform unusable, blurry vehicle images into valuable data for both personal and business applications, like in parking garages and speed cameras. It uses advanced ML models to detect vehicles and license plates in images, followed by rigorous image processing for enhanced clarity and number isolation. With sophisticated digit recognition, AutoBot accurately reads license plate numbers.</p>

<p>What sets AutoBot apart is its ability to address complex challenges. It connects to a comprehensive Israeli vehicle database for up-to-date, precise information. In cases of extremely blurred images or obscured license plate numbers, AutoBot uniquely infers missing characters by analyzing available characters, vehicle color, and database information. This iterative approach ensures the accurate identification of the complete license plate number and corresponding vehicle details.</p>

<p>Additionally, AutoBot features a sophisticated chatbot interface for interactive user queries and assistance. This chatbot component is capable of understanding and responding to various user inputs, enhancing the system's accessibility and user experience.</p>

<h2>Key Features of AutoBot</h2>
<ul>
    <li><b>Advanced Image Processing:</b> Employs techniques like edge detection, histogram equalization, and adaptive thresholding for image clarity and number isolation.</li>
    <li><b>Machine Learning-Based Vehicle and License Plate Detection:</b> Utilizes pre-trained YOLO model for accurate detection of vehicles and their license plates.</li>
    <li><b>Color Detection:</b> Integrates a neural network model for determining the color of detected vehicles.</li>
    <li><b>Digit Recognition:</b> Applies connected components analysis for extracting and recognizing digits on license plates.</li>
    <li><b>Image Rotation and Alignment:</b> Corrects skewed images to facilitate accurate digit recognition.</li>
    <li><b>Robust Error Handling:</b> Designed to manage multiple detected objects and ambiguities in image processing.</li>
    <li><b>Intelligent Chatbot Interface:</b> Offers interactive communication for user assistance and query handling, utilizing an AI-based natural language processing model.</li>
</ul>

<h2>Technologies and Libraries Used</h2>
<ul>
    <li><b>Python:</b> The primary programming language for implementing the system.</li>
    <li><b>OpenCV (cv2):</b> A comprehensive library used for real-time computer vision and image processing.</li>
    <li><b>NumPy:</b> Essential for numerical operations on arrays and matrices, supporting various data manipulations.</li>
    <li><b>Pillow (PIL):</b> An image processing library utilized for manipulating and processing image data.</li>
    <li><b>Keras:</b> A high-level neural networks API for loading and using neural network models, running on top of TensorFlow.</li>
    <li><b>Matplotlib:</b> For data visualization and presentation, particularly useful in debugging and presentation modes.</li>
    <li><b>torch:</b> A deep learning library providing a wide range of functionalities for neural networks.</li>
    <li><b>json:</b> For parsing and manipulating JSON data.</li>
    <li><b>tempfile:</b> Used for creating temporary files during intermediate stages of image processing.</li>
    <li><b>math:</b> Provides basic mathematical functions and operations, enhancing various computational aspects.</li>
</ul>

<h2>Environments Used</h2>
<ul>
    <li><b>General Python Environment:</b> The code is designed to be run in a standard Python environment.</li>
    <li><b>Integration with External Databases and Systems:</b> Capable of connecting to vehicle databases for extended information retrieval.</li>
</ul>

<h2>Algorithms Used</h2>
<ul>
    <li><b>YOLO (You Only Look Once):</b> For real-time object detection.</li>
    <li><b>Canny Edge Detection:</b> Used for detecting edges in images.</li>
    <li><b>Hough Line Transform:</b> For line detection in image processing.</li>
    <li><b>Connected Components Analysis:</b> Employed for identifying and isolating digits on license plates.</li>
    <li><b>Image Thresholding Techniques:</b> Including adaptive thresholding for image binarization.</li>
    <li><b>Natural Language Processing:</b> Utilized in the chatbot interface for understanding and responding to user queries.</li>
</ul>

