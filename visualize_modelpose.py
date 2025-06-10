import cv2
import json
import os

IMAGE_FOLDER = "modelposes"
JSON_FOLDER = "modelposes_json"

files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]
files.sort()

index = 0
total = len(files)

def draw_landmarks(image, keypoints, angles, label):
    h, w = image.shape[:2]
    for name, coord in keypoints.items():
        x = int(coord["x"] * w)
        y = int(coord["y"] * h)
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    y_offset = 20
    for name, angle in angles.items():
        text = f"{name}: {angle if angle is not None else 'n/a'}°"
        cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        y_offset += 15
    cv2.putText(image, f"Pose: {label} ({index+1}/{total})", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

while True:
    if total == 0:
        print("❌ Geen JSON-bestanden gevonden.")
        break

    filename = files[index]
    json_path = os.path.join(JSON_FOLDER, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data["filename"]
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Kan afbeelding niet laden: {image_path}")
        index = (index + 1) % total
        continue

    display = image.copy()
    display = draw_landmarks(display, data["keypoints"], data.get("angles", {}), data["label"])
    cv2.imshow("Modelpose Viewer", display)
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif key in [ord("q"), 83, 2555904]:
        index = (index + 1) % total
    elif key in [ord("d"), 81, 2424832]:
        index = (index - 1 + total) % total

cv2.destroyAllWindows()