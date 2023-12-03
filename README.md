<h1>AutoBot - Advanced Vehicle Identification System</h1>
<h2><a href="https://youtu.be/NCGko0xxi4U?feature=shared">Click here for a video demonstration of AutoBot</a></h2>
<h2>Description</h2>
<p>AutoBot leverages Computer Vision and AI to transform unusable, blurry vehicle images into valuable data for both personal and business applications, like in parking garages and speed cameras. It uses advanced ML models to detect vehicles and license plates in images, followed by rigorous image processing for enhanced clarity and number isolation. With sophisticated digit recognition, AutoBot accurately reads license plate numbers.</p>

<p>What sets AutoBot apart is its ability to address complex challenges. It connects to a comprehensive Israeli vehicle database for up-to-date, precise information. In cases of extremely blurred images or obscured license plate numbers, AutoBot uniquely infers missing characters by analyzing available characters, vehicle color, and database information. This iterative approach ensures the accurate identification of the complete license plate number and corresponding vehicle details.</p>

<h2>Key Features of AutoBot</h2>
<ul>
    <li><b>Advanced Image Processing:</b> Employs techniques like edge detection, histogram equalization, and adaptive thresholding for image clarity and number isolation.</li>
    <li><b>Machine Learning-Based Vehicle and License Plate Detection:</b> Utilizes pre-trained YOLO model for accurate detection of vehicles and their license plates.</li>
    <li><b>Color Detection:</b> Integrates a neural network model for determining the color of detected vehicles.</li>
    <li><b>Digit Recognition:</b> Applies connected components analysis for extracting and recognizing digits on license plates.</li>
    <li><b>Image Rotation and Alignment:</b> Corrects skewed images to facilitate accurate digit recognition.</li>
    <li><b>Robust Error Handling:</b> Designed to manage multiple detected objects and ambiguities in image processing.</li>
</ul>

<h2>Technologies and Libraries Used</h2>
<ul>
    <li><b>Python:</b> The primary programming language for implementing the system.</li>
    <li><b>OpenCV (cv2):</b> A comprehensive library used for real-time computer vision and image processing.</li>
    <li><b>NumPy:</b> Essential for numerical operations on arrays and matrices, supporting various data manipulations.</li>
    <li><b>Pillow (PIL):</b> An image processing library utilized for manipulating and processing image data.</li>
    <li><b>Keras:</b> A high-level neural networks API for loading and using neural network models, running on top of TensorFlow.</li>
    <li><b>Matplotlib:</b> For data visualization and presentation, particularly useful in debugging and presentation modes.</li>
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
</ul>


<h2>Parameters</h2>
<ul>
    <li><b>Pivot Date:</b> A critical date used to segment the transaction data for comparative analysis. Transactions are analyzed separately for periods before and after this date to understand changes over time as well as filter them.</li>
    <li><b>Threshold Percent for Missing Data:</b> A parameter defining the acceptable percentage of missing data in city transactions. Cities exceeding this threshold are excluded from the analysis to maintain data integrity.</li>
    <li><b>Transaction Category:</b> Focuses the analysis on a specific category of transactions, allowing for more targeted insights into spending patterns within that category.</li>
    <li><b>Clustering Sensitivity (EPS Value for DBSCAN):</b> Determines the sensitivity of the DBSCAN clustering algorithm. A crucial parameter for identifying clusters with varying densities in the data.</li>
    <li><b>Minimum Samples for DBSCAN:</b> Specifies the minimum number of samples (or total weight) in a neighborhood for a point to be considered a core point. This parameter is integral to the DBSCAN algorithm's ability to form clusters.</li>
    <li><b>Number of Components for Isomap:</b> Dictates the number of dimensions to which the data is reduced, influencing the granularity of the dimensionality reduction process.</li>
    <li><b>Number of Neighbors in Nearest Neighbors Analysis:</b> Sets the number of neighbors to consider in the nearest neighbors analysis, impacting the identification of closely related observations within the data.</li>
</ul>
