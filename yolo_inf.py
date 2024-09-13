import os
from ultralytics import YOLO

model = YOLO('models/object-detection.pt')

DATA_DIR = 'input_videos'

res = []

for v_name in os.listdir(DATA_DIR):
    res.append(model.predict(os.path.join(DATA_DIR, v_name), save=True))

# Check if working
print(res[0][100])

print("\n\n\n")

for box in res[0][100].boxes:
    print(box)