import cv2
import mediapipe as mp
import numpy as np
import json
import time
import streamlit as st
import requests

FEEDBACK_INTERVAL = 0.5
MAX_ANGLE_DIFF = 45
JOINT_RADIUS = 10
LINE_THICKNESS = 5
LINE_SEGMENTS = 10
LLM_ENABLED = True

def call_local_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"⚠️ AI-feedback kon niet worden opgehaald: {str(e)}"

def generate_friendly_feedback(joint_name, user_angle, model_angle, priority, description, pose_name):
    prompt = (
        f"Je bent een vriendelijke yoga-coach.\n"
        f"De gebruiker probeert de pose \"{pose_name}\" uit te voeren.\n\n"
        f"Beschrijving van de pose: {description}\n\n"
        f"De ideale hoek van het gewricht {joint_name} is {round(model_angle)} graden, "
        f"maar de gebruiker behaalde {round(user_angle)} graden. "
        f"Dit gewricht is belangrijk voor deze pose (prioriteit {priority}).\n\n"
        f"Formuleer een korte, begrijpelijke en vriendelijke feedbackzin over hoe de gebruiker dit kan verbeteren."
    )
    return call_local_llm(prompt)

def midpoint(p1, p2):
    return tuple(int((p1[i] + p2[i]) / 2) for i in range(2))

def gradient_color(norm):
    if not isinstance(norm, (int, float)) or np.isnan(norm):
        return (200, 200, 200)
    if norm <= 0.5:
        r = int(255 * (norm * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (norm - 0.5) * 2))
    return (0, g, r)

def draw_gradient_line(img, pt1, pt2, color1, color2, segments=10, thickness=6):
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        x1 = int(pt1[0] * (1 - t1) + pt2[0] * t1)
        y1 = int(pt1[1] * (1 - t1) + pt2[1] * t1)
        x2 = int(pt1[0] * (1 - t2) + pt2[0] * t2)
        y2 = int(pt1[1] * (1 - t2) + pt2[1] * t2)
        c = [int(color1[j] * (1 - t1) + color2[j] * t1) for j in range(3)]
        cv2.line(img, (x1, y1), (x2, y2), c, thickness)
def draw_gradient_line(img, pt1, pt2, color1, color2, segments=10, thickness=6):
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        x1 = int(pt1[0] * (1 - t1) + pt2[0] * t1)
        y1 = int(pt1[1] * (1 - t1) + pt2[1] * t1)
        x2 = int(pt1[0] * (1 - t2) + pt2[0] * t2)
        y2 = int(pt1[1] * (1 - t2) + pt2[1] * t2)
        c = [int(color1[j] * (1 - t1) + color2[j] * t1) for j in range(3)]
        cv2.line(img, (x1, y1), (x2, y2), c, thickness)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def run_streamlit_feedback(modelpose_path, streamlit_placeholder, hold_time=30, feedback_text_area=None, timer_area=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

    with open(modelpose_path) as f:
        model_json = json.load(f)
    model_angles = model_json["angles"]
    model_priority = model_json.get("priority", {})
    model_head_turn = model_json.get("head_turn")
    model_description = model_json.get("description", "")
    model_label = model_json.get("label", "deze pose")

    cap = cv2.VideoCapture(0)
    last_score_update = 0
    last_score = None
    last_joint_colors = {}
    feedback_lines = []

    start_time = time.time()
    while cap.isOpened() and (time.time() - start_time < hold_time):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if timer_area:
            remaining = int(hold_time - (time.time() - start_time))
            timer_area.markdown(f"⏳ Houd vast... {remaining}s")

        head_turn = None  # Reset bij elk frame

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            pose_px = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))}
            virtual_joints = {}

            if 11 in pose_px and 12 in pose_px:
                virtual_joints["neck"] = midpoint(pose_px[11], pose_px[12])
            if 7 in pose_px and 8 in pose_px:
                virtual_joints["head"] = midpoint(pose_px[7], pose_px[8])
            if 23 in pose_px and 24 in pose_px:
                virtual_joints["hip_center"] = midpoint(pose_px[23], pose_px[24])

            try:
                left = lm[mp_pose.PoseLandmark.LEFT_EAR.value].x
                right = lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x
                nose = lm[mp_pose.PoseLandmark.NOSE.value].x
                center = (left + right) / 2
                head_turn = round(nose - center, 3)
            except:
                head_turn = None

            def pt(x):
                return pose_px.get(x) if isinstance(x, int) else virtual_joints.get(x)

            current_time = time.time()
            update_feedback = (current_time - last_score_update) >= FEEDBACK_INTERVAL
            feedback_lines = [] if update_feedback else None

            raw_colors = {}
            total_diff = 0
            counted = 0

            ANGLE_DEFINITIONS = {
                "left_elbow": [11, 13, 15], "right_elbow": [12, 14, 16],
                "left_knee": [23, 25, 27], "right_knee": [24, 26, 28],
                "neck": [11, "neck", 12], "hip": [23, "hip_center", 24],
                "head": ["neck", "head", 0],
                "left_shoulder": [23, 11, 13], "right_shoulder": [24, 12, 14],
                "left_wrist": [13, 15, 19], "right_wrist": [14, 16, 20],
                "left_ankle": [25, 27, 31], "right_ankle": [26, 28, 32],
            }

            for name, (a, b, c) in ANGLE_DEFINITIONS.items():
                pa, pb, pc = pt(a), pt(b), pt(c)
                if pa and pb and pc:
                    angle = np.degrees(np.arccos(np.clip(np.dot(np.subtract(pa, pb), np.subtract(pc, pb)) /
                                                          (np.linalg.norm(np.subtract(pa, pb)) * np.linalg.norm(np.subtract(pc, pb))), -1.0, 1.0)))
                    model = model_angles.get(name + "_angle")
                    if model is None:
                        continue
                    diff = abs(angle - model)
                    norm = min(diff / MAX_ANGLE_DIFF, 1.0)
                    color = gradient_color(norm)

                    for joint in [a, b, c]:
                        if joint not in raw_colors:
                            raw_colors[joint] = []
                        raw_colors[joint].append(color)

                    if update_feedback:
                        prio = model_priority.get(name + "_angle", 5)
                        if prio >= 7 and diff >= 10:
                            if LLM_ENABLED:
                                msg = generate_friendly_feedback(name, angle, model, prio, model_description, model_label)
                            else:
                                msg = f"{name}_angle: {round(angle)}° vs {round(model)}° (prio {prio})"
                            feedback_lines.append(msg)
                        total_diff += norm
                        counted += 1

            if update_feedback and counted > 0:
                last_score = max(0, int((1 - total_diff / counted) * 100))
                last_score_update = current_time
                last_joint_colors = {
                    j: tuple(int(np.mean([c[i] for c in clist])) for i in range(3))
                    for j, clist in raw_colors.items()
                }

                if head_turn is not None and model_head_turn is not None:
                    diff = abs(head_turn - model_head_turn)
                    norm = min(diff / 0.2, 1.0)
                    last_joint_colors["head"] = gradient_color(norm)
                    prio = model_priority.get("head_turn", 5)
                    if prio >= 7 and diff >= 0.05:
                        feedback_lines.append(f"head_turn offset: {round(diff, 3)} (prio {prio})")

            joint_colors = {str(k): v for k, v in last_joint_colors.items()}

            links = [
                ("head", "neck"), ("neck", 11), ("neck", 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
                ("neck", "hip_center"),
                ("hip_center", 23), ("hip_center", 24),
                (23, 25), (25, 27), (24, 26), (26, 28),
            ]

            for p1_id, p2_id in links:
                p1, p2 = pt(p1_id), pt(p2_id)
                if p1 and p2:
                    c1 = joint_colors.get(str(p1_id), (200, 200, 200))
                    c2 = joint_colors.get(str(p2_id), (200, 200, 200))
                    draw_gradient_line(frame, p1, p2, c1, c2, LINE_SEGMENTS, LINE_THICKNESS)

            visible_joints = set([
                "head", "neck", "hip_center",
                11, 12, 13, 14, 15, 16,
                23, 24, 25, 26, 27, 28
            ])

            for idx in visible_joints:
                coord = pt(idx)
                if coord:
                    color = joint_colors.get(str(idx), (255, 255, 255))
                    cv2.circle(frame, coord, JOINT_RADIUS, color, -1)

            if last_score is not None:
                cv2.putText(frame, f"Pose Score: {last_score}%", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        streamlit_placeholder.image(frame, channels="RGB")

        if feedback_text_area:
            with feedback_text_area.container():
                if feedback_lines is not None and feedback_lines:
                    st.markdown("### Live Feedback")
                    for line in feedback_lines:
                        st.markdown(f"- {line}")
                elif feedback_lines is not None:
                    if last_score is not None and last_score >= 85:
                        st.info("✅ Geen significante afwijkingen gevonden bij belangrijke gewrichten.")
                    else:
                        st.warning("⚠️ Pose komt nog niet voldoende overeen met het model.")

        time.sleep(1 / 30)

    cap.release()
    return last_score if last_score is not None else 0