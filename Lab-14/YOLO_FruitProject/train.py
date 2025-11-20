from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Завантажуємо модель
    model = YOLO("yolo11n.pt")

    # 2. Вказуємо шлях до вашого ВЖЕ ВИПРАВЛЕНОГО файлу
    # (Ми не використовуємо Roboflow тут, щоб він не перезаписав файл)
    yaml_file = "/home/tmarchenko/Code/University/AI Systems /Lab-14/YOLO_FruitProject/My-First-Project-1/data.yaml"

    # 3. Тренування
    print("Починаємо тренування на CPU...")
    model.train(data=yaml_file, epochs=20, imgsz=640, device='cpu')
    