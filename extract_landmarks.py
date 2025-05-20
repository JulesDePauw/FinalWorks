import cv2
import mediapipe as mp
import os
import json
import math
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

INPUT_FOLDER = "modelposes"
OUTPUT_FOLDER = "modelposes_json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calculate_angle(a, b, c):
    ba = [a['x'] - b['x'], a['y'] - b['y']]
    bc = [c['x'] - b['x'], c['y'] - b['y']]
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    magnitude = (math.hypot(*ba) * math.hypot(*bc))
    if magnitude == 0:
        return 0.0
    angle_rad = math.acos(dot_product / magnitude)
    return math.degrees(angle_rad)

def calculate_head_turn(keypoints):
    try:
        left = keypoints["left_ear"]["x"]
        right = keypoints["right_ear"]["x"]
        nose = keypoints["nose"]["x"]
        center = (left + right) / 2
        return round(nose - center, 3)
    except KeyError:
        return None

def get_pose_description_and_priority(label):
    pose_name = label.replace("_", " ").replace("-", " ").replace("Pose", "").strip().lower()
    system_msg = "Je bent een yogacoach."
    user_prompt = f"""
Provide a JSON object for the yoga pose '{pose_name}' with two fields:
1. \"description\": a short description of the pose (max 2 sentences)
2. \"joint_priority\": a dictionary with joint angles (e.g. 'left_knee_angle', 'head_turn') and a priority score from 1 to 10.

Respond with only valid JSON. Do not include explanations, formatting notes, or any other text.
"""
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )
    try:
        print("⚡ LLM response:", response.choices[0].message.content)
        content = response.choices[0].message.content.strip().strip("`").strip("json").strip()
        data = json.loads(content)
        return data.get("description", ""), data.get("joint_priority", {})
    except Exception as e:
        print("⚠️ Fout bij JSON parsing:", e)
        print("🧾 Response was:", response.choices[0].message.content)
        return "", {}

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(path)
    if image is None:
        print(f"❌ Fout bij openen van {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"⚠️ Geen landmarks gevonden in {filename}")
        continue

    landmarks = results.pose_landmarks.landmark

    keypoints = {}
    for i, lm in enumerate(landmarks):
        name = mp_pose.PoseLandmark(i).name.lower()
        keypoints[name] = {
            "x": lm.x,
            "y": lm.y
        }

    angles = {}
    def angle_of(a, b, c, label):
        try:
            angles[label] = round(calculate_angle(keypoints[a], keypoints[b], keypoints[c]), 1)
        except KeyError:
            angles[label] = None

    angle_of("left_shoulder", "left_elbow", "left_wrist", "left_elbow_angle")
    angle_of("left_hip", "left_knee", "left_ankle", "left_knee_angle")
    angle_of("right_shoulder", "right_elbow", "right_wrist", "right_elbow_angle")
    angle_of("right_hip", "right_knee", "right_ankle", "right_knee_angle")
    angle_of("left_elbow", "left_shoulder", "left_hip", "left_shoulder_angle")
    angle_of("right_elbow", "right_shoulder", "right_hip", "right_shoulder_angle")
    angle_of("left_shoulder", "left_hip", "left_knee", "left_hip_angle")
    angle_of("right_shoulder", "right_hip", "right_knee", "right_hip_angle")
    angle_of("left_knee", "left_ankle", "left_foot_index", "left_ankle_angle")
    angle_of("right_knee", "right_ankle", "right_foot_index", "right_ankle_angle")

    head_turn = calculate_head_turn(keypoints)

    label = os.path.splitext(filename)[0]
    description, joint_priority = get_pose_description_and_priority(label)

    output = {
        "filename": filename,
        "label": label,
        "description": str(description),
        "priority": dict(joint_priority),
        "keypoints": keypoints,
        "angles": angles,
        "head_turn": head_turn
    }

    out_path = os.path.join(OUTPUT_FOLDER, f"{label}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ JSON aangemaakt: {out_path}")
