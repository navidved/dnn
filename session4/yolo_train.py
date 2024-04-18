from ultralytics import YOLO


model = YOLO('yolov8n.pt')
results = model.train(data='/home/navid/Projects/teaching/DNN/session4/ds/car_ds/data.yaml', epochs=100, imgsz=640) 