"""Phase 2: continuous tracking with persistent state, faithful rendering.

Decodes the original video once, tracks across the whole stream (clips are NOT
state boundaries), logs per-frame results keyed by frame_idx/pts, and renders
the annotated video from the ORIGINAL frames.

Usage: python run_tracking.py in.mp4 out.mp4 results.jsonl [weights.pt]
Requires: pip install ultralytics opencv-python
"""
import json
import sys

import cv2
from ultralytics import YOLO


def main(src, dst, log, weights="yolo11s.pt"):
    model = YOLO(weights)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    with open(log, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            r = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
            tracks = []
            if r.boxes is not None and r.boxes.id is not None:
                for b, tid, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                        r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
                    tracks.append({"id": tid, "cls": model.names[k], "conf": round(c, 3), "xyxy": b})
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{model.names[k]} {tid}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            f.write(json.dumps({"frame_idx": idx, "pts": round(pts, 4), "tracks": tracks}) + "\n")
            out.write(frame)
            idx += 1
    cap.release(); out.release()
    print(f"frames written: {idx}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
