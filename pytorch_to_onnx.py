from ultralytics import YOLO

model = YOLO('models/yolo26n_rail_final.pt')
model.export(format='onnx', imgsz=1536)
print('✅ Exported to ONNX!')

