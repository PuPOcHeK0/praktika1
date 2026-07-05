"""Финальная система контроля безопасности."""
import cv2
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict
from ultralytics import YOLO


class SafetyMonitor:
    def __init__(self, weights="runs/detect/yolov8s_hardhat/weights/best.pt",
                 conf=0.5, violations_dir="violations"):
        self.model = YOLO(weights)
        self.conf = conf
        self.violations_dir = Path(violations_dir)
        self.violations_dir.mkdir(exist_ok=True)
        self.stats = defaultdict(int)
        self.log = []

    def process_frame(self, frame, frame_id=0):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        violations = []

        persons = [b for b in results.boxes if int(b.cls) == 2]

        for person in persons:
            px1, py1, px2, py2 = person.xyxy[0].tolist()
            helmet_found = False
            head_found = False

            for b in results.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                cx, cy = (bx1+bx2)/2, (by1+by2)/2
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    if int(b.cls) == 0:
                        helmet_found = True
                    elif int(b.cls) == 1:
                        head_found = True

            if head_found and not helmet_found:
                violations.append({
                    "person_bbox": [px1, py1, px2, py2],
                    "confidence": float(person.conf),
                })

        if violations:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.violations_dir / f"violation_{frame_id}_{ts}.jpg"
            annotated = self._draw(frame, results, violations)
            cv2.imwrite(str(out_path), annotated)
            self.stats["violations"] += len(violations)
            self.log.append({
                "frame": frame_id,
                "time": ts,
                "count": len(violations),
                "file": str(out_path),
            })

        self.stats["frames"] += 1
        self.stats["persons"] += len(persons)
        return violations

    def _draw(self, frame, results, violations):
        colors = [(0, 255, 0), (0, 0, 255), (255, 200, 0)]
        names = ["helmet", "head", "person"]
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls = int(b.cls)
            color = colors[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{names[cls]} {b.conf:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if violations:
            cv2.putText(frame, f"VIOLATIONS: {len(violations)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return frame

    def report(self):
        report = {
            "total_frames": self.stats["frames"],
            "total_persons": self.stats["persons"],
            "total_violations": self.stats["violations"],
            "violation_rate": self.stats["violations"] / max(1, self.stats["persons"]),
            "log": self.log[-20:],
        }
        (self.violations_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False))
        return report