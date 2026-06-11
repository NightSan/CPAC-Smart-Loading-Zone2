from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # เริ่มใหม่
    
    results = model.train(
        data="cement-mixer-truckv5-1/data.yaml",
        imgsz=640,
        epochs=300,
        patience=50,
        batch=8,
        workers=0,
        name="concretemix_v6_new",
    )
    
    print("✅ เทรนเสร็จแล้ว!")
    print("weights: runs/detect/concretemix_v6_new/weights/best.pt")