import sys
from pathlib import Path

print("=" * 60, flush=True)
print("СКАЧИВАНИЕ ДАТАСЕТА через huggingface_hub", flush=True)
print("=" * 60, flush=True)

# Проверяем, что huggingface_hub работает
try:
    import huggingface_hub
    print(f" huggingface_hub: {huggingface_hub.__version__}", flush=True)
except ImportError:
    print(" pip install huggingface_hub", flush=True)
    sys.exit(1)

from huggingface_hub import snapshot_download

OUTPUT_DIR = Path("hardhat_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n Целевая папка: {OUTPUT_DIR.resolve()}", flush=True)
print(" Скачивание займёт 5-15 минут (~1.5 ГБ)...\n", flush=True)

try:
    path = snapshot_download(
        repo_id="keremberke/hard-hat-detection",
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR),
        resume_download=True,
    )
    print(f"\n Скачано в: {path}", flush=True)
    
    # Считаем файлы
    n_files = sum(1 for _ in OUTPUT_DIR.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f" Файлов: {n_files}, размер: {total_size / (1024**3):.2f} ГБ", flush=True)
    
except Exception as e:
    print(f"\n Ошибка: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60, flush=True)
print("ГОТОВО! Переходи к конвертации в YOLO-формат.", flush=True)
print("=" * 60, flush=True)