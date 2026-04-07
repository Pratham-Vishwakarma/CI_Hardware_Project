from collections import Counter
import serial
import time

WINDOW_SIZE = 10
WINDOW_TITLE = "Cognitive Study Assistant"


def infer_cognitive_state(emotion_scores, recent_emotions):
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    sorted_scores = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1]
    score_gap = top_score - second_score

    surprise_score = emotion_scores.get("surprise", 0.0)
    fear_score = emotion_scores.get("fear", 0.0)
    sad_score = emotion_scores.get("sad", 0.0)
    neutral_score = emotion_scores.get("neutral", 0.0)
    happy_score = emotion_scores.get("happy", 0.0)

    if sad_score >= 0.35:
        return "Tired"

    if surprise_score + fear_score >= 0.55:
        return "Distracted"

    if score_gap <= 0.12 and (surprise_score >= 0.15 or fear_score >= 0.15):
        return "Confused"

    if dominant_emotion in {"neutral", "happy"} and (neutral_score + happy_score >= 0.55):
        return "Focused"

    if recent_emotions.count("surprise") >= 3:
        return "Distracted"

    if recent_emotions.count("sad") >= 3:
        return "Tired"

    return "Confused"


def get_state_color(state):
    if state == "Focused":
        return (80, 220, 80)
    if state == "Distracted":
        return (0, 215, 255)
    if state == "Confused":
        return (0, 140, 255)
    if state == "Tired":
        return (255, 140, 140)
    return (255, 255, 255)

def send_state_to_arduino(state, arduino):
    if state == "Focused":
        arduino.write(b'F')

    elif state == "Distracted":
        arduino.write(b'D')

    elif state == "Confused":
        arduino.write(b'C')

    elif state == "Tired":
        arduino.write(b'T')

def run_study_assistant():
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import cv2
    from fer.fer import FER

    arduino = serial.Serial("COM3", 9600)
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    detector = FER()

    emotion_history = []
    state_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_emotions(frame)

        if detections:
            emotion_scores = detections[0]["emotions"]
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)

            emotion_history.append(dominant_emotion)
            if len(emotion_history) > WINDOW_SIZE:
                emotion_history.pop(0)

            inferred_state = infer_cognitive_state(emotion_scores, emotion_history)

            state_history.append(inferred_state)
            if len(state_history) > WINDOW_SIZE:
                state_history.pop(0)

            stable_state = Counter(state_history).most_common(1)[0][0]
            send_state_to_arduino(stable_state, arduino)
            display_color = get_state_color(stable_state)

            cv2.putText(
                frame,
                f"Study State: {stable_state}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                display_color,
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (180, 180, 180),
                2,
            )

        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_study_assistant()
