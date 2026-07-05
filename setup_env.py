import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Создаёт и настраивает виртуальное окружение"""
    
    venv_dir = Path("venv")
    
    # 1. Создаём venv, если его нет
    if not venv_dir.exists():
        print(" Создаю виртуальное окружение...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print(" venv создан")
    else:
        print(" venv уже существует")
    
    # 2. Определяем путь к pip в venv
    if os.name == "nt":  # Windows
        pip_path = venv_dir / "Scripts" / "pip.exe"
        python_path = venv_dir / "Scripts" / "python.exe"
    else:  # Linux/Mac
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
    
    # 3. Обновляем pip
    print("\n Обновляю pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # 4. Устанавливаем PyTorch с CUDA 12.1
    print("\n Устанавливаю PyTorch с CUDA 12.1...")
    subprocess.run([
        str(pip_path), "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)
    
    # 5. Устанавливаем остальные зависимости
    print("\n Устанавливаю зависимости...")
    requirements = [
        "ultralytics",           # YOLOv8 + RT-DETR
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tqdm",
        "pillow",
        "pycocotools",
        "datasets",              # Hugging Face
        "huggingface_hub"
    ]
    
    subprocess.run([str(pip_path), "install"] + requirements, check=True)
    
    # 6. Клонируем YOLOv5
    print("\n Клонирую YOLOv5...")
    if not Path("yolov5").exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
        subprocess.run([str(pip_path), "install", "-r", "yolov5/requirements.txt"], check=True)
    
    # 7. Сохраняем requirements.txt
    print("\n Сохраняю requirements.txt...")
    subprocess.run([str(pip_path), "freeze", ">", "requirements.txt"], shell=True)
    
    print("\n" + "="*50)
    print(" Окружение готово!")
    print("="*50)
    print(f"\n Путь к Python: {python_path}")
    print(f" Путь к pip: {pip_path}")
    print("\n Для активации venv в терминале:")
    if os.name == "nt":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")

if __name__ == "__main__":
    setup_environment()