from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # เริ่มใหม่
    
    results = model.train(
        data="cement-mixer-truckv10-7/data.yaml",
        imgsz=320,
        epochs=300,
        patience=50,
        batch=8,
        workers=0,
        name="concretemix_v8",
    )
    
    print("✅ เทรนเสร็จแล้ว!")
    print("weights: runs/detect/concretemix_v8/weights/best.pt")