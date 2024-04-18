from ultralytics import YOLO

model = YOLO('/home/navid/Projects/teaching/DNN/session4/runs/detect/train2/weights/best.pt')

prd = model.predict(source='/home/navid/Projects/teaching/DNN/session4/ds/test/test2.jpg', conf=0.25, save=True, project="session4")

print(prd)