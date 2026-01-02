# K-Pop Fancam AI Generator 🎬

Ứng dụng AI tự động tạo fancam cho thần tượng K-Pop từ video concert/performance, sử dụng YOLOv8 để tracking và deep learning để nhận diện.

## 📋 Tính năng

- 🎯 **Auto Tracking**: Tự động phát hiện và theo dõi người được chọn trong video
- 🤖 **Face Recognition**: Nhận diện khuôn mặt bằng deep learning
- 🎨 **Color Matching**: So sánh màu trang phục để tăng độ chính xác
- 🔍 **Smart Zoom**: Tự động zoom và crop theo tỷ lệ 9:16 (vertical video)
- ⚡ **CPU Optimized**: Tối ưu hóa để chạy trên CPU, không cần GPU
- 🎵 **Audio Sync**: Giữ nguyên âm thanh gốc từ video

## 🛠️ Yêu cầu hệ thống

- Python 3.8 - 3.11
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- CPU: Intel Core i5 hoặc tương đương
- Dung lượng: ~5GB cho models và dependencies

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd fancam_cpu
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Kiểm tra cài đặt
```bash
# Test YOLO model
python test_yolo.py

# Test GPU availability (optional)
python test_gpu.py
```

## 🚀 Chạy ứng dụng

### Khởi động server
```bash
python server.py
```

Server sẽ chạy tại: **http://localhost:5000**

### Mở trình duyệt
Truy cập: `http://localhost:5000` để sử dụng giao diện web

## 📖 Hướng dẫn sử dụng

### Bước 1: Upload Video
- Click vào khu vực **"Drag and drop your video here"**
- Hoặc kéo thả file video vào khu vực này
- Format hỗ trợ: MP4, AVI, MOV

### Bước 2: Detect People
- Click nút **"Detect People"** để phát hiện người trong video
- Hệ thống sẽ hiển thị ảnh với các ID được đánh số
- Ghi nhớ ID của người bạn muốn tạo fancam

### Bước 3: Nhập Target ID
- Nhập ID của người bạn muốn theo dõi vào ô **"Target Person ID"**
- Ví dụ: nếu muốn theo dõi người có ID 2, nhập `2`

### Bước 4: Upload Reference Images (Optional)
- **Face Images**: Upload ảnh khuôn mặt của người đó (1-3 ảnh)
- **Outfit Images**: Upload ảnh trang phục (1-3 ảnh)
- Ảnh càng rõ nét, độ chính xác càng cao

### Bước 5: Chọn Zoom Level
- Điều chỉnh mức zoom: 1.0x - 4.0x
- **1.0x**: Toàn thân (recommended)
- **1.5x**: Nửa người
- **2.0x+**: Close-up

### Bước 6: Generate Fancam
- Click nút **"Generate Fancam"**
- Đợi quá trình xử lý (2-10 phút tùy độ dài video)
- Video output sẽ hiển thị khi hoàn thành

### Bước 7: Download
- Click **"Download Video"** để tải về
- Video được lưu với format MP4, codec H.264

## 📁 Cấu trúc dự án

```
fancam_cpu/
├── server.py              # Flask server backend
├── main.py                # Core AI processing logic
├── app.js                 # Frontend JavaScript
├── fancam_ui.html         # UI template
├── styles.css             # Styling
├── requirements.txt       # Python dependencies
├── yolov8n.pt            # YOLOv8 nano model
├── yolov8m.pt            # YOLOv8 medium model (optional)
├── test_yolo.py          # YOLO test script
├── test_gpu.py           # GPU check script
└── flagged/              # Temporary storage
```

## 🔧 Cấu hình

### Thay đổi model (nếu có GPU)
Trong [`main.py`](main.py) dòng 28-32:

```python
# Sử dụng CPU (mặc định)
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
DEVICE_STR = "cpu"
model = YOLO("yolov8n.pt")

# Nếu có GPU, bỏ comment:
# DEVICE_STR = "0"  # GPU ID
# model = YOLO("yolov8m.pt")
```

### Điều chỉnh tham số tracking
Trong [`main.py`](main.py) dòng 34-40:

```python
CONFIDENCE_THRESHOLD = 0.3      # Ngưỡng confidence YOLO
FACE_SIM_THRESHOLD = 0.65       # Ngưỡng tương đồng khuôn mặt
MAX_CENTER_DISTANCE = 500       # Khoảng cách tối đa giữa frames
MAX_LOST_FRAMES = 60            # Số frames tối đa mất tracking
```

## 🐛 Troubleshooting

### Lỗi: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Lỗi: "OpenCV not found"
```bash
pip install opencv-python opencv-contrib-python
```

### Lỗi: "Memory Error"
- Giảm độ phân giải video đầu vào
- Tăng RAM hệ thống
- Đóng các ứng dụng khác

### Video output bị lệch âm thanh
- Kiểm tra codec của video gốc
- Đảm bảo moviepy đã cài đặt đầy đủ:
```bash
pip install moviepy[optional]
```

### Tracking không chính xác
1. Upload thêm reference images (face + outfit)
2. Tăng `FACE_SIM_THRESHOLD` trong [`main.py`](main.py)
3. Giảm `MAX_CENTER_DISTANCE` để tracking chặt chẽ hơn

## 📊 Performance

- **Video 1080p, 3 phút**: ~5-7 phút processing (CPU)
- **Video 4K, 5 phút**: ~15-20 phút processing (CPU)
- **GPU acceleration**: Nhanh hơn 3-5 lần

## 🔒 Privacy & Security

- Tất cả xử lý được thực hiện **local** trên máy bạn
- Không upload video lên cloud
- Temporary files tự động xóa sau khi xử lý
- Reference images chỉ lưu trong session

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📝 License

MIT License - Free to use for personal and commercial projects

## 👤 Author

**Fancam AI Team**
- GitHub: [Your GitHub]
- Email: [Your Email]

## 🙏 Credits

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [OpenCV](https://opencv.org/)
- [MoviePy](https://zulko.github.io/moviepy/)
- [Flask](https://flask.palletsprojects.com/)

## 📸 Screenshots

### Main Interface
![Main UI](docs/screenshot_main.png)

### Detection Result
![Detection](docs/screenshot_detection.png)

### Processing
![Processing](docs/screenshot_processing.png)

---

**⭐ If you find this useful, please star the repository!**