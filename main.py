import cv2
from ultralytics import YOLO
from vars import IMAGE_PATH, MODEL, RESULT_PATH, INFO_INPUT

def _find_count_of_person(image, boxes, confidences, class_ids):
    num_people = 0

    for i in range(len(boxes)):
        if int(class_ids[i]) == 0:
            num_people += 1
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'Person: {confidences[i]:.2f}'
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return num_people
def find_person_on_image(image_path, result_path, model):
    image = cv2.imread(image_path)
    model = YOLO(model)
    results = model(image_path, conf=0.2)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    output_image = image.copy()

    cv2.putText(output_image, f'Number of people: {_find_count_of_person(output_image, boxes, confidences, class_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(result_path, output_image)
    cv2.imshow('Image with People Detected', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    added_path = input(INFO_INPUT)
    full_path = IMAGE_PATH + added_path
    find_person_on_image(full_path, RESULT_PATH, MODEL)