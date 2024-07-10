import cv2
import numpy as np
from ultralytics import YOLO
from typing import Any, Optional, Tuple

from vars import IMAGE_PATH, MODEL, RESULT_PATH, INFO_INPUT


def _euclidean_distance(point1: Optional[list], point2: Optional[list]) -> Any:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def _find_count_of_person(image: Any, boxes: Any, confidences: Any, class_ids: Any) -> Tuple:
    num_people = 0
    people_centers = []
    people_boxes = []

    for i in range(len(boxes)):
        if int(class_ids[i]) == 0:
            num_people += 1
            x1, y1, x2, y2 = map(int, boxes[i])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            people_centers.append((center_x, center_y))
            people_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f'Person: {confidences[i]:.2f}'
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)

    return num_people, people_centers, people_boxes

def check_for_helmet(image: Any, box: Any) -> bool:
    x1, y1, x2, y2 = box
    head_region = image[y1:y1 + (y2 - y1) // 2, x1:x2]
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # lower_green = np.array([35, 100, 100])
    # upper_green = np.array([85, 255, 255])
    # mask_green = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(mask_orange) > 0:
        return True
    return False


def find_groups(image: Any, people_centers: Optional[list], people_boxes: Optional[list]) -> None:
    groups = []
    visited = set()

    for i, center in enumerate(people_centers):
        if i not in visited:
            group = [i]
            visited.add(i)
            for j, other_center in enumerate(people_centers):
                if j != i and _euclidean_distance(center, other_center) < 100:
                    group.append(j)
                    visited.add(j)
            groups.append(group)

    for group in groups:
        if len(group) > 1:
            x1 = min(people_boxes[i][0] for i in group)
            y1 = min(people_boxes[i][1] for i in group)
            x2 = max(people_boxes[i][2] for i in group)
            y2 = max(people_boxes[i][3] for i in group)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f'Group of {len(group)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def find_person_on_image(image_path: str, result_path: str, model: str) -> None:
    image = cv2.imread(image_path)
    model = YOLO(model)
    results = model(image_path, conf=0.2)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    output_image = image.copy()

    num_people, people_centers, people_boxes = _find_count_of_person(output_image, boxes, confidences, class_ids)

    cv2.putText(output_image, f'Number of people: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    find_groups(output_image, people_centers, people_boxes)

    num_with_helmet = 0
    num_without_helmet = 0

    for box in people_boxes:
        if check_for_helmet(output_image, box):
            num_with_helmet += 1
        else:
            num_without_helmet += 1

    cv2.putText(output_image, f'With Helmet: {num_with_helmet}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_image, f'Without Helmet: {num_without_helmet}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imwrite(result_path, output_image)
    cv2.imshow('Image with People, Groups, and Helmets Detected', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    added_path = input(INFO_INPUT)
    full_path = IMAGE_PATH + added_path
    find_person_on_image(full_path, RESULT_PATH, MODEL)
