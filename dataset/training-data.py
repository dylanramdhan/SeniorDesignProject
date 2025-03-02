from ultralytics import YOLO

# loading in YOLOv8 model (starting with a pre-trained one)
model = YOLO("yolov8n.pt")  # can use "yolov8s.pt" or "yolov8m.pt" for better accuracy

# training the model on your labeled dataset
model.train(data="/Users/dylanramdhan/Documents/GitHub/PersonalRepo/SeniorDesignProject/dataset/dataset.yaml", epochs=50, imgsz=640)

print("✅ Training complete! Best model saved in 'runs/train/exp/weights/best.pt'")