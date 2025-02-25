import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util, visualization_utils as viz_utils
import matplotlib.pyplot as plt
# %matplotlib inline

# Load the saved model
SAVED_MODEL_PATH = "D://Preme//github//HatOK//model//research//exported_model//saved_model"

detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)

# Load the label map
LABEL_MAP_PATH = "D://Preme//github//HatOK//model//ssd_mobilenet_v2//train//label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Function to load and preprocess an image
def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image_rgb

# Path to the test image
IMAGE_PATH = r'D:\Preme\test helmet\test helmet4.png'
image_np = load_image_into_numpy_array(IMAGE_PATH)

# Run inference
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension
detections = detect_fn(input_tensor)

# Process the detections
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# Convert detection classes to integers
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

# Visualization
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=50,
    min_score_thresh=0.5,  # Adjust this threshold as needed
    agnostic_mode=False
)
# Extract the height and width of the image
image_height, image_width, _ = image_np.shape

# Iterate through detection boxes, classes, and scores
for i in range(num_detections):
    # Only consider detections above the threshold (e.g., 0.5)
    if detections['detection_scores'][i] >= 0.5:
        # Get normalized coordinates
        ymin, xmin, ymax, xmax = detections['detection_boxes'][i]

        # Convert normalized coordinates to pixel values
        xmin_pixel = int(xmin * image_width)
        xmax_pixel = int(xmax * image_width)
        ymin_pixel = int(ymin * image_height)
        ymax_pixel = int(ymax * image_height)
        print(f"  Bounding Box (pixels): [{xmin_pixel}, {ymin_pixel}, {xmax_pixel}, {ymax_pixel}]")

# Display the image with detections
plt.figure(figsize=(12, 8))
plt.imshow(image_np)
plt.axis('off')
plt.show()
plt.savefig('detection_output.png')  # Save the figure to a file