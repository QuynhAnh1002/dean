import cv2
import mediapipe as mp
import time
from pathlib import Path
import pygame


def draw_warning(frame, text="LOCK IN TWIN"):
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        text,
        (w // 4, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1.4,
        (0, 0, 255),
        3
    )


def main():
    # ====== CONFIG ======
    TIMER = 1.5
    LOOKING_DOWN_THRESHOLD = 0.4
    DEBOUNCE_THRESHOLD = 0.5

    VIDEO_PATH = Path("assets/skyrim-skeleton.mp4")
    AUDIO_PATH = Path("assets/skyrim-skeleton.mp3")

    if not VIDEO_PATH.exists():
        print("❌ Không tìm thấy video")
        return
    if not AUDIO_PATH.exists():
        print("❌ Không tìm thấy audio")
        return

    # ====== INIT ======
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    cam = cv2.VideoCapture(0)

    video_cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    pygame.mixer.init()

    doomscroll_start = None
    video_playing = False

    # ====== MAIN LOOP ======
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face_mesh.process(rgb)

        current = time.time()

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            left_eye = [lm[145], lm[159]]
            right_eye = [lm[374], lm[386]]

            l_iris = lm[468]
            r_iris = lm[473]

            l_ratio = (l_iris.y - left_eye[1].y) / (left_eye[0].y - left_eye[1].y + 1e-6)
            r_ratio = (r_iris.y - right_eye[1].y) / (right_eye[0].y - right_eye[1].y + 1e-6)
            avg_ratio = (l_ratio + r_ratio) / 2

            is_looking_down = avg_ratio < (
                DEBOUNCE_THRESHOLD if video_playing else LOOKING_DOWN_THRESHOLD
            )

            cv2.putText(
                frame,
                f"Ratio: {avg_ratio:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if not is_looking_down else (0, 0, 255),
                2
            )

            if is_looking_down:
                if doomscroll_start is None:
                    doomscroll_start = current

                elapsed = current - doomscroll_start
                cv2.rectangle(
                    frame,
                    (10, 60),
                    (int(10 + min(elapsed / TIMER, 1) * 150), 75),
                    (0, 0, 255),
                    -1
                )

                if elapsed >= TIMER and not video_playing:
                    video_playing = True
                    pygame.mixer.music.load(str(AUDIO_PATH))
                    pygame.mixer.music.play(-1)  # loop audio
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            else:
                doomscroll_start = None
                if video_playing:
                    video_playing = False
                    pygame.mixer.music.stop()
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cv2.destroyWindow("DOOMSCROLL WARNING")

        # ====== PLAY VIDEO ======
        if video_playing:
            ret_v, vframe = video_cap.read()
            if not ret_v:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_v, vframe = video_cap.read()

            vframe = cv2.resize(vframe, (w, h))
            draw_warning(vframe)
            cv2.imshow("DOOMSCROLL WARNING", vframe)
            cv2.waitKey(delay)
        else:
            cv2.imshow("DOOMSCROLL CHECKER", frame)
            if cv2.waitKey(1) == 27:
                break

    # ====== CLEANUP ======
    pygame.mixer.music.stop()
    cam.release()
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
