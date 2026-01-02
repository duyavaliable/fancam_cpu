from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")

print("Running detection...")
results = model("test.jpg")  # thay bằng 1 file ảnh bất kỳ

print("Done!")
