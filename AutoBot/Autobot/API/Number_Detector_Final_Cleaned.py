import math
import sys
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Color_Detector_Done
from Digit_Detector_Final import detect

presentation_mode = 0


def get_inner_outer(boxes, indexes):
    counter = 0
    upper_left_x = math.inf
    upper_left_y = math.inf
    box_list = []

    for i in range(len(boxes)):
        if i in indexes:
            counter += 1
            box_list.append(boxes[i])
        x, y, w, h = boxes[i]
        upper_left_x = min(upper_left_x, x)
        upper_left_y = min(upper_left_y, y)

    if len(box_list) != 2:
        print(f"Sorry, please take a photo of only one car. Detected {len(box_list)} boxes.")
        return box_list, 3
    else:
        inner_box = 3
        x1, y1, w1, h1 = box_list[0]
        x2, y2, w2, h2 = box_list[1]
        if (x1 < x2) & (y1 < y2) & (x1 + w1 > x2 + w2) & (y1 + h1 > y2 + h2):
            inner_box = 0
        if (x1 > x2) & (y1 > y2) & (x1 + w1 < x2 + w2) & (y1 + h1 < y2 + h2):
            inner_box = 1
    if inner_box == 3:
        print('Could not find inner box')
    return box_list, inner_box == 3


def find_skew_angle(image, edge_threshold1=30, edge_threshold2=200, hough_threshold=70, min_line_length=200,
                    max_line_gap=40, horizontal_thresh=45):
    N = 4
    edges = cv2.Canny(image, edge_threshold1, edge_threshold2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if lines is None:
        return None

    lengths_and_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi

        if -horizontal_thresh <= angle <= horizontal_thresh:
            lengths_and_angles.append(((x1, y1, x2, y2), line_length, angle))

    if not lengths_and_angles:
        return None

    lengths_and_angles.sort(key=lambda x: -x[1])
    if presentation_mode == 1:
        image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2), _, _ in lengths_and_angles[:N]:
            cv2.line(image_copy, (x1, y1), (x2, y2), (10, 255, 10), 10)
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title('Detected Lines we align with')
        plt.show()

    average_angle = np.mean([angle for _, _, angle in lengths_and_angles[:N]])
    return average_angle


def fix_angle(img):
    angle = find_skew_angle(img)
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def pre_procc_plate_img_bin(image):
    plate_img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_img_gray = cv2.equalizeHist(plate_img_gray)
    plate_img_gray = cv2.GaussianBlur(plate_img_gray, (7, 7), 0)
    _, plate_img_binary = cv2.threshold(plate_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return plate_img_binary


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_plate_hist_angle_bin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plate_img_gray = cv2.equalizeHist(gray)
    plate_img_gray = cv2.resize(plate_img_gray, (1600, 490))
    plate_img_gray = cv2.GaussianBlur(plate_img_gray, (3, 3), 0)

    # _, plate_img_binary = cv2.threshold(plate_img_gray, 95, 255, cv2.THRESH_BINARY_INV)

    plate_img_binary = cv2.bitwise_not(plate_img_gray)
    block_size = 191  # Size of a pixel neighborhood that is used to calculate a threshold value
    c = 1  # Constant subtracted from the mean
    plate_img_binary = cv2.adaptiveThreshold(plate_img_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                             block_size, c)

    # plate_img_inverted = cv2.bitwise_not(plate_img_binary)
    return fix_angle(plate_img_binary)


def prepare_and_detect(componentMask_cropped):
    temp_componentMask_cropped_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
    resized_componentMask_cropped = componentMask_cropped.copy()
    old_size = resized_componentMask_cropped.shape[:2]
    ratio = float(65) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_componentMask_cropped = cv2.resize(resized_componentMask_cropped, (new_size[1], new_size[0]))
    delta_w = 39 - new_size[1]
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    resized_componentMask_cropped = cv2.copyMakeBorder(resized_componentMask_cropped, 0, 0, left, right,
                                                       cv2.BORDER_CONSTANT, value=color)
    # if presentation_mode == 1:
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(142), plt.imshow(cv2.cvtColor(resized_componentMask_cropped, cv2.COLOR_BGR2RGB)), plt.title('new size')
    #     plt.show()
    cv2.imwrite(temp_componentMask_cropped_file, resized_componentMask_cropped)
    ans = detect(temp_componentMask_cropped_file)
    return ans


def calculate_max_distance_and_boxes(original_boxes, indices):
    max_distance = 0
    max_distance_indices = None
    for i in range(1, len(indices)):
        previous_index = indices[i - 1]
        current_index = indices[i]
        previous_bbox = original_boxes[previous_index]
        current_bbox = original_boxes[current_index]
        distance = current_bbox[0] - previous_bbox[2]

        if distance > max_distance:
            max_distance = distance
            max_distance_indices = (previous_index, current_index)

    return max_distance_indices


def plot_components_at_original_positions(original_boxes, component_images, indices, max_distance_indices):
    if not original_boxes:
        return
    max_x = max(box[2] for box in original_boxes)
    max_y = max(box[3] for box in original_boxes)
    canvas = np.zeros((max_y, max_x))

    for i in indices:
        x1, y1, x2, y2 = original_boxes[i]
        width = x2 - x1
        height = y2 - y1
        component = cv2.resize(component_images[i], (width, height))
        canvas[y1:y2, x1:x2] = component

    x1, _, x2, _ = original_boxes[max_distance_indices[0]]
    x3, _, _, y3 = original_boxes[max_distance_indices[1]]
    canvas[0:y3, x2:x3 - 1] = 255
    return 0


def init(image_path):
    net = cv2.dnn.readNet("Trained_Models/Yolov4/yolov4-ANPR.weights", "Trained_Models/Yolov4/yolov4-ANPR.cfg")
    output_layers = net.getUnconnectedOutLayersNames()
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    box_list, flag = get_inner_outer(boxes, [indexes[0], indexes[1]])
    plate_box = box_list[flag]
    car_box = box_list[1 - flag]

    return img, car_box, plate_box


def fully_process_plate(plate_img):
    plate_img_hist = preprocess_plate_hist_angle_bin(plate_img)
    plate_img_hist = np.bitwise_not(plate_img_hist)

    plate_img_hist = cv2.resize(plate_img_hist, (2300, 600))
    # kernel = np.ones((3, 3), np.uint8)
    # plate_img_hist = cv2.dilate(plate_img_hist, kernel, iterations=3)
    # plate_img_hist = cv2.erode(plate_img_hist, kernel, iterations=5)

    if presentation_mode == 1:
        plt.figure(figsize=(10, 5))
        plt.subplot(141), plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)), plt.title('Original Plate')
        plt.subplot(142), plt.imshow(cv2.cvtColor(plate_img_hist, cv2.COLOR_BGR2RGB)), plt.title('Processed Plate')

    return plate_img_hist


def get_components(min_pixel_threshold, numLabels, labels, stats):
    answers = []
    original_boxes = []
    component_images = []
    for i in range(0, numLabels):
        if 55000 > stats[i, cv2.CC_STAT_AREA] >= min_pixel_threshold and stats[i, cv2.CC_STAT_AREA] < stats[
            0, cv2.CC_STAT_AREA] and stats[i, cv2.CC_STAT_WIDTH] < 400:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            componentMask = (labels == i).astype("uint8") * 255
            componentMask_cropped = componentMask[y:y + h, x:x + w]
            top, bottom, left, right = [11] * 4
            componentMask_cropped = cv2.copyMakeBorder(componentMask_cropped, top, bottom, left, right,
                                                       cv2.BORDER_CONSTANT, value=0)
            componentMask_cropped = cv2.resize(componentMask_cropped, (60, 100))
            digit, acc = prepare_and_detect(componentMask_cropped)
            if acc > 0.8:
                component_images.append(componentMask_cropped)
                original_boxes.append((x, y, x + w, y + h))
                answers.append(digit)
            else:
                component_images.append(componentMask_cropped)
                original_boxes.append((x, y, x + w, y + h))
                answers.append('x')
    return answers, original_boxes, component_images


def sort_answer(answers, original_boxes, component_images):
    indices = sorted(range(len(answers)), key=lambda i: original_boxes[i][0])
    final_string = ""
    max_distance = 0
    for i in range(len(indices)):
        index = indices[i]
        digit = answers[index]
        original_bbox = original_boxes[index]
        final_string += str(digit)
        if i > 0:
            previous_index = indices[i - 1]
            previous_bbox = original_boxes[previous_index]
            current_bbox = original_bbox
            distance = current_bbox[0] - previous_bbox[2]
            if distance > max_distance:
                max_distance = distance

    max_distance_indices = calculate_max_distance_and_boxes(original_boxes, indices)
    plot_components_at_original_positions(original_boxes, component_images, indices, max_distance_indices)
    if presentation_mode == 1:
        plt.figure(figsize=(1 * len(component_images), 4))
        for i, cmp in enumerate(component_images):
            plt.subplot(1, len(component_images), i + 1)
            plt.imshow(cmp, cmap='gray')
            plt.axis('off')
        plt.show()
    return final_string


def start(img, car_box, plate_box):
    xp, yp, wp, hp = plate_box
    xc, yc, wc, hc = car_box
    plate_img = img[yp:yp + hp, xp:xp + wp]
    car_img = img[yc:yc + hc, xc:xc + wc]
    detected_color = Color_Detector_Done.get_color(car_img)
    cv2.rectangle(img, (xp, yp), (xp + wp, yp + hp), (0, 255, 0), 2)
    cv2.rectangle(img, (xc, yc), (xc + wc, yc + hc), (0, 255, 0), 2)
    plate_img_hist = fully_process_plate(plate_img)
    (numLabels, labels, stats, _) = cv2.connectedComponentsWithStats(plate_img_hist, 8, cv2.CV_32S)
    min_pixel_threshold = 14000
    answers, original_boxes, component_images = get_components(min_pixel_threshold, numLabels, labels, stats)
    if presentation_mode == 1:
        plt.figure(figsize=(1 * len(component_images), 4))
        for i, cmp in enumerate(component_images):
            plt.subplot(1, len(component_images), i + 1)
            plt.imshow(cmp, cmap='gray')
            plt.axis('off')
        plt.show()
    custom_str = sort_answer(answers, original_boxes, component_images)

    if len(custom_str) > 0:
        cv2.putText(img, custom_str, (xp, yp - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(plate_img, (0, 0), (plate_img.shape[1], plate_img.shape[0]), (0, 255, 0), 2)

    cv2.putText(img, detected_color, (xc, yc - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if presentation_mode == 1:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    print(custom_str)


if __name__ == "__main__":
    presentation_mode = 1
    path = sys.argv[1]
    img, car_box, plate_box = init(path)
    start(img, car_box, plate_box)
