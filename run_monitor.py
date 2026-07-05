"""Запуск системы мониторинга на видео."""
import cv2
import argparse
from safety_monitor import SafetyMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="0",
                        help="Путь к видео или ID камеры (0)")
    parser.add_argument("--weights", type=str,
                        default="runs/detect/yolov8s_hardhat/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    monitor = SafetyMonitor(weights=args.weights, conf=args.conf)

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f" Не удалось открыть видео: {source}")
        return

    print("=" * 60)
    print("СИСТЕМА КОНТРОЛЯ БЕЗОПАСНОСТИ")
    print("=" * 60)
    print(f"Видео: {source}")
    print("Нажмите 'q' для выхода")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        violations = monitor.process_frame(frame, frame_id)

        if violations:
            print(f"[Frame {frame_id}] ⚠️  Нарушений: {len(violations)}")

        cv2.imshow("Safety Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_id % 100 == 0:
            print(f"[Frame {frame_id}] Статистика: {dict(monitor.stats)}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    report = monitor.report()
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Всего кадров: {report['total_frames']}")
    print(f"Обнаружено людей: {report['total_persons']}")
    print(f"Нарушений (без каски): {report['total_violations']}")
    print(f"Коэффициент нарушений: {report['violation_rate']:.2%}")
    print(f"\n Отчёт: violations/report.json")


if __name__ == "__main__":
    main()