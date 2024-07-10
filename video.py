import cv2
import torch
from torchvision import models, transforms
import numpy as np
import time

model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

input_video_path = "video.mp4"
output_video_path = "output_video.mp4"

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Ошибка")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

track_id = 0
track_dict = {}
total_time = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Остановка рендеринга")
        break

    start_time = time.time()

    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img)[0]

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    person_boxes = boxes[labels == 1]
    person_scores = scores[labels == 1]

    for i, box in enumerate(person_boxes):
        if person_scores[i] > 0.5:
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            if (x, y, w, h) not in track_dict.values():
                track_dict[track_id] = (x, y, w, h)
                track_id += 1
            person_id = [k for k, v in track_dict.items() if v == (x, y, w, h)][0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    count = len(person_boxes)
    cv2.putText(frame, f"Количество: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

    end_time = time.time()

    frame_time = end_time - start_time
    total_time += frame_time
    frame_count += 1

    if frame_count % 10 == 0:
        print(f"Average time per frame after {frame_count} frames: {total_time / frame_count:.4f} seconds")

cap.release()
out.release()
cv2.destroyAllWindows()


average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Final average time per frame: {average_time_per_frame:.4f} seconds")

estimated_total_time = average_time_per_frame * total_frames
print(f"Estimated total processing time: {estimated_total_time:.2f} seconds")

print(f"Обработка видео завершена. Результат сохранен в {output_video_path}")