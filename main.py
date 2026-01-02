from ultralytics import YOLO
import cv2
import numpy as np
import os
import datetime
# from deepface import DeepFace
from numpy.linalg import norm
from moviepy.editor import VideoFileClip, AudioFileClip 
import time
import torch
import gc
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tắt log TF thừa


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             # Chỉ cho phép TF lấy dung lượng cần thiết, không chiếm hết 4GB
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


# Tải model (YOLOv8n) và cấu hình device
# DEVICE = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu' 
# model = YOLO("yolov8m.pt") 
# LOG_FILE = "fancam_error.log"
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
DEVICE_STR = "cpu"
model = YOLO("yolov8n.pt")
LOG_FILE = "fancam_error.log"
CONFIDENCE_THRESHOLD = 0.3
EMA_ALPHA = 0.005
MAX_IOU_AREA_THRESHOLD = 2000
MAX_CENTER_DISTANCE = 500  # ✅ TĂNG TỪ 300 → 500 (chấp nhận nhảy xa hơn)
FACE_SIM_THRESHOLD = 0.65
MAX_LOST_FRAMES = 60  # ✅ THÊM MỚI: Ngưỡng fast-forward
FAST_FORWARD_INTERVAL = 5  # ✅ THÊM MỚI: Check mỗi 5 frames


# --- HÀM TIỆN ÍCH: GHI LỖI VÀO TỆP LOG ---
def log_error(step, error_message, level="ERROR"):
    """Ghi lỗi hoặc cảnh báo vào tệp log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] - STEP: {step} - MESSAGE: {error_message}\n"
    print(log_entry.strip()) # In ra console để debug nhanh
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Không thể ghi log vào tệp {LOG_FILE}: {e}")


# Bổ sung cảnh báo nếu dùng CPU
# try:
#     # BUỘC THIẾT BỊ LÀ CUDA:0 (GPU 1 - Quadro P600)
#     # Nếu có lỗi, nó sẽ bị bắt ngay lập tức.
#     DEVICE_STR = "cuda:0" 
    
#     # Khởi tạo model và tải nó vào GPU
#     model = YOLO("yolov8n.pt") 
#     # Bắt buộc chuyển model sang GPU ngay sau khi tải
#     model.to(DEVICE_STR) 
    
#     print(f"\n✅ THÀNH CÔNG: Model đã được tải vào {DEVICE_STR}\n")
    
# except Exception as e:
#     # Nếu thất bại, chúng ta sẽ buộc dùng CPU và ghi lỗi
#     DEVICE_STR = "cpu"
#     model = YOLO("yolov8n.pt") 
#     log_error("Model Setup", f"LỖI NGHIÊM TRỌNG KHI KHỞI TẠO CUDA. Model đang chạy trên CPU: {e}")
    
#     print("\n---------------------------------------------------------")
#     print("!!! KHỞI TẠO CUDA THẤT BẠI. CHƯƠNG TRÌNH CHẠY TRÊN CPU !!!")
#     print("---------------------------------------------------------\n")

def is_blur(image, threshold=60):
    """Kiểm tra ảnh có bị nhòe không để tránh Re-ID nhầm."""
    if image is None or image.size == 0: return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tính độ biến thiên Laplacian - cách nhanh nhất đo độ nét
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold

def enhance_face(face_img):
    """Tăng chất lượng vùng mặt bằng thuật toán truyền thống (không tốn GPU)."""
    if face_img is None or face_img.size == 0: return face_img
    
    # 1. Resize về chuẩn Facenet 160x160 để model xử lý tốt nhất
    face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
    
    # 2. Tăng độ sắc nét bằng Unsharp Masking
    gaussian = cv2.GaussianBlur(face_img, (0, 0), 2.0)
    enhanced = cv2.addWeighted(face_img, 1.5, gaussian, -0.5, 0)
    
    # 3. Cân bằng ánh sáng cục bộ (CLAHE) giúp mặt rõ nét hơn
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_embedding(image_data):
    """Hàm trích xuất embedding đã tối ưu hóa chất lượng ảnh đầu vào."""
    from deepface import DeepFace
    try:
        # Nhận dữ liệu ảnh (có thể là đường dẫn string hoặc mảng numpy từ crop)
        img = cv2.imread(image_data) if isinstance(image_data, str) else image_data
        
        if img is None: return None
        
        # Bước 1: Bỏ qua nếu ảnh quá mờ
        if is_blur(img): return None
        
        # Bước 2: Tăng cường chất lượng ảnh mặt
        processed_face = enhance_face(img)
        
        # Bước 3: Trích xuất đặc trưng với DeepFace
        embedding = DeepFace.represent(
            img_path=processed_face,
            model_name="Facenet512",
            enforce_detection=False,
            align=True,           # Tự động căn chỉnh mắt/mũi để Re-ID chuẩn hơn
            detector_backend='opencv' # Dùng backend nhanh nhất cho GPU 4GB
        )[0]["embedding"]
        
        return np.array(embedding)
    except Exception as e:
        log_error("Face Embedding", f"Lỗi trích xuất embedding: {e}")
        return None

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)

def handle_upload(new_files, current_files):
    """Thêm tệp mới (new_files) vào danh sách tệp hiện tại (current_files)."""
    if new_files is None:
        # Trường hợp người dùng hủy bỏ hộp thoại chọn tệp
        # Trả về current_files cho State, và paths cho Output
        output_paths = [f.name for f in (current_files or [])]
        return current_files, output_paths
    
    if not isinstance(new_files, list):
        new_files = [new_files]
        
    # [QUAN TRỌNG]: Lưu trữ toàn bộ đối tượng file trong State
    updated_files = (current_files or []) + new_files
    
    # [SỬA LỖI]: Trả về CHUỖI ĐƯỜNG DẪN cho Output hiển thị (file_output)
    output_paths = [f.name for f in updated_files]
    
    # Trả về danh sách đối tượng file cho State, và danh sách path cho Output
    return updated_files, output_paths

# --- HÀM 1: PHÁT HIỆN VÀ CHỌN NGƯỜI BAN ĐẦU (Không đổi) ---
def initial_detection(video_path):
    if not video_path:
        log_error("Initial Detection", "Không có đường dẫn video đầu vào.")
        return None, "Vui lòng tải lên một tệp video.", []
    
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
    except Exception as e:
        log_error("Initial Detection", f"Lỗi khi mở/đọc video: {e}")
        return None, "Lỗi khi mở/đọc video.", []
    
    if not ret:
        log_error("Initial Detection", "Không thể đọc khung hình đầu tiên (ret=False).")
        return None, "Không thể đọc khung hình từ video.", []

    try:
        # Phát hiện người (person - class 0)
        results = model(frame, classes=0, device=DEVICE_STR, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
    except Exception as e:
        log_error("Initial Detection", f"Lỗi trong quá trình YOLOv8 phát hiện: {e}")
        return None, "Lỗi trong quá trình phát hiện đối tượng.", []

    # ... (Các logic tính toán boxes, areas, detections vẫn giữ nguyên)
    if len(boxes) == 0:
        log_error("Initial Detection", "Không tìm thấy người nào trong khung hình đầu tiên.")
        return None, "Không tìm thấy người nào trong khung hình đầu tiên.", []

    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    
    detections = []
    for i, box in enumerate(boxes):
        detections.append({
            'index': i + 1,
            'box': box,
            'area': areas[i]
        })

    detections.sort(key=lambda x: x['area'], reverse=True)
    default_target_box = detections[0]['box']
    
    sample_frame = frame.copy()
    info_list = []
    for det in detections:
        box = det['box'].astype(int)
        index = det['index']
        cv2.rectangle(sample_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(sample_frame, f'ID: {index}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        info_list.append(f"ID {index} - Diện tích: {int(det['area'])}")

    temp_img_path = "temp_detection.jpg"
    cv2.imwrite(temp_img_path, sample_frame)

    return temp_img_path, "Đã phát hiện người trong khung hình đầu tiên. Vui lòng chọn ID người bạn muốn tạo fancam.", "\n".join(info_list)

def get_color_histogram(frame, box):
    """Trích xuất màu sắc TOÀN THÂN để tăng độ nhận diện khi đổi hướng."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    
    # ✅ LẤY TOÀN BỘ BODY (30% - 90% chiều cao)
    y1_body = y1 + int((y2 - y1) * 0.3)
    y2_body = y1 + int((y2 - y1) * 0.9)
    
    crop = frame[max(0, y1_body):min(h_img, y2_body), max(0, x1):min(w_img, x2)]
    if crop.size == 0: return None
    
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # ✅ TĂNG SỐ BIN (16x16 thay vì 12x12)
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_colors(hist1, hist2):
    """So sánh độ tương đồng màu sắc (0 đến 1)."""
    if hist1 is None or hist2 is None: return 0
    # Sử dụng HISTCMP_CORREL (Tương quan) để có độ chính xác cao
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def linear_interpolate_center(history, max_gap):
    """
    Nội suy tuyến tính các vị trí bị mất dấu (lost) trong tracking_history.
    """
    interpolated_centers = np.array([[h[1], h[2]] for h in history])
    i = 0
    while i < len(history):
        if not history[i][3]:  # Tìm điểm bắt đầu của khoảng trống (is_found=False)
            start_i = i - 1
            if start_i < 0:
                i += 1
                continue
                
            end_i = i
            while end_i < len(history) and not history[end_i][3]:
                end_i += 1
            
            if end_i == len(history):
                break
            
            gap_length = end_i - start_i
            
            # Chỉ nội suy nếu khoảng trống đủ nhỏ (tạo chuyển động camera mượt)
            if gap_length <= max_gap:
                center_A = interpolated_centers[start_i]
                center_B = interpolated_centers[end_i]
                
                for j in range(start_i + 1, end_i):
                    t = (j - start_i) / gap_length
                    interpolated_centers[j] = center_A + (center_B - center_A) * t
                i = end_i
            else:
                # Khoảng trống quá lớn, giữ nguyên vị trí cuối cùng được biết
                center_A = interpolated_centers[start_i]
                for j in range(start_i + 1, end_i):
                     interpolated_centers[j] = center_A
                i = end_i 
        i += 1
    return interpolated_centers





def get_color_histogram(frame, box):
    """Trích xuất màu sắc TOÀN THÂN để tăng độ nhận diện khi đổi hướng."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    
    # ✅ LẤY TOÀN BỘ BODY (30% - 90% chiều cao)
    y1_body = y1 + int((y2 - y1) * 0.3)
    y2_body = y1 + int((y2 - y1) * 0.9)
    
    crop = frame[max(0, y1_body):min(h_img, y2_body), max(0, x1):min(w_img, x2)]
    if crop.size == 0: return None
    
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # ✅ TĂNG SỐ BIN (16x16 thay vì 12x12)
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def process_fancam(video_path, target_id_str, ref_face_paths, ref_color_paths, zoom_level):
    # --- 1. KHỞI TẠO & ĐỊNH NGHĨA PATH (Đã sửa lỗi mất path) ---
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    
    output_path_temp = "temp_silent_fancam.avi"
    MAX_JUMP_DIST = 0.05  # Tỷ lệ phần trăm tối đa của chiều rộng video 
    log_error("Setup", "Bắt đầu tiến trình: Ưu tiên Mặt (20-60%) + Màu trung tâm (30-60%) + Vị trí (10-20%)", "INFO")

    # --- 2. TRÍCH XUẤT MẪU ---
    ref_embedding = None
    if ref_face_paths:
        all_embs = [extract_embedding(f.name) for f in ref_face_paths if extract_embedding(f.name) is not None]
        if all_embs: ref_embedding = np.mean(all_embs, axis=0)

    ref_hist = None
    if ref_color_paths:
        sample_img = cv2.imread(ref_color_paths[0].name)
        res_c = model(sample_img, classes=0, conf=0.3, verbose=False)[0]
        if len(res_c.boxes) > 0:
            ref_hist = get_color_histogram(sample_img, res_c.boxes.xyxy.cpu().numpy()[0])

    # --- 3. VIDEO & KHỞI TẠO ID CHUẨN TẠI FRAME 1 ---
    cap_info = cv2.VideoCapture(video_path)
    width, height = int(cap_info.get(3)), int(cap_info.get(4))
    fps, total_frames = cap_info.get(5) or 30, int(cap_info.get(7))
    ret, frame_init = cap_info.read()
    cap_info.release()

    # Quét Re-ID ngay frame 1 để khớp ID tracker với ID mục tiêu
    # res_i = model.track(frame_init, persist=True, classes=0, verbose=False)[0]
    res_i = model.track(frame_init, persist=True, classes=0, device=DEVICE_STR, imgsz=640, half=False, verbose=False)[0]
    boxes_i = res_i.boxes.xyxy.cpu().numpy()
    ids_i = res_i.boxes.id.cpu().numpy().astype(int) if res_i.boxes.id is not None else []
    
    current_target_id = -1
    for box, tid in zip(boxes_i, ids_i):
        f_crop = frame_init[int(box[1]):int(box[1]+(box[3]-box[1])*0.5), int(box[0]):int(box[2])]
        emb = extract_embedding(enhance_face(f_crop))
        if emb is not None and ref_embedding is not None:
            if cosine_similarity(ref_embedding, emb) > 0.7: 
                current_target_id = tid
                break

    if current_target_id == -1:
        try:
            # Chuyển input của ní thành số nguyên
            target_idx = int(target_id_str) - 1 
            
            # Kiểm tra xem index có nằm trong danh sách không (tránh lỗi out of bounds)
            if 0 <= target_idx < len(ids_i):
                current_target_id = ids_i[target_idx]
                log_error("Init", f"Dùng ID người dùng chọn: {current_target_id}", "INFO")
            else:
                # Nếu nhập số quá lớn, mặc định lấy người đầu tiên (index 0)
                current_target_id = ids_i[0]
                log_error("Init", f"ID chọn nằm ngoài danh sách ({len(ids_i)} người), lấy ID {current_target_id} mặc định.", "WARN")
        except:
            return "Lỗi: ID nhập vào không hợp lệ.", None

    idx_init = np.where(ids_i == current_target_id)[0][0]
    prev_cx, prev_cy = (boxes_i[idx_init][0]+boxes_i[idx_init][2])/2, (boxes_i[idx_init][1]+boxes_i[idx_init][3])/2
    kalman.statePost = np.array([[prev_cx], [prev_cy], [0], [0]], np.float32)

    # --- 4. PASS 1: SMART TRACKING ---
    tracking_history = []
    frame_count = 0
    lost_counter = 0  # ✅ BIẾN ĐẾM SỐ FRAME MẤT DẤU LIÊN TỤC
    MAX_LOST_FRAMES = 60  # ✅ NGƯỠNG CHUYỂN SANG FAST-FORWARD (60 frames = 2 giây ở 30fps)
    FAST_FORWARD_INTERVAL = 5  # ✅ KHI FAST-FORWARD, CHỈ CHECK MỖI 5 FRAMES
    
    results = model.track(source=video_path, tracker="bytetrack.yaml", persist=True, 
                          imgsz=384, classes=0, device=DEVICE_STR, stream=True, verbose=False)

    for res in results:
        frame_count += 1
        
        # ✅ HIỂN THỊ TIẾN ĐỘ MỖI 50 FRAMES
        if frame_count % 50 == 0:
            log_error("Progress", f"⏱️ Frame {frame_count}/{total_frames} | Lost: {lost_counter}f", "INFO")
        
        frame = res.orig_img
        pred = kalman.predict()
        
        all_b = res.boxes.xyxy.cpu().numpy() if res.boxes.id is not None else []
        all_ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else []

        found_box = None
        
        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 1: KIỂM TRA ID HIỆN TẠI (LUÔN CHẠY MỌI FRAME)
        # ═══════════════════════════════════════════════════════════════
        if current_target_id in all_ids:
            idx = np.where(all_ids == current_target_id)[0][0]
            curr_box = all_b[idx]
            curr_cx, curr_cy = (curr_box[0]+curr_box[2])/2, (curr_box[1]+curr_box[3])/2
            dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            
            if dist < MAX_CENTER_DISTANCE:  # ✅ DÙNG 500px thay vì 300px
                found_box = curr_box
                lost_counter = 0  # ✅ TÌM THẤY → RESET ĐẾM
            else: 
                log_error("Motion", f"F{frame_count}: Nhảy quá xa ({int(dist)}px)", "WARN")
                lost_counter += 1  # ✅ NHẢY XA QUÁ → TĂNG ĐẾM

        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 2: XỬ LÝ KHI MẤT DẤU (CHỈ CHẠY KHI found_box = None)
        # ═══════════════════════════════════════════════════════════════
        if found_box is None:
            lost_counter += 1
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 🚀 MODE FAST-FORWARD (CHỈ KÍCH HOẠT KHI lost_counter > 60)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if lost_counter > MAX_LOST_FRAMES:
                # ✅ BỎ QUA 4/5 FRAMES (chỉ check frame 61, 66, 71, 76...)
                if (frame_count - MAX_LOST_FRAMES) % FAST_FORWARD_INTERVAL != 0:
                    tracking_history.append((frame_count, pred[0][0], pred[1][0], False))
                    continue  # ← NHẢY QUA FRAME NÀY, KHÔNG CHẠY RE-ID
                
                log_error("FastForward", f"F{frame_count}: Tìm lại (mất {lost_counter}f)...", "WARN")
                
                # ✅ QUÉT TOÀN BỘ KHUNG HÌNH - CHỈ DÙNG MÀU SẮC
                best_score, best_id, temp_box = 0, None, None
                for box, tid in zip(all_b, all_ids):
                    c_s = compare_colors(ref_hist, get_color_histogram(frame, box)) if ref_hist is not None else 0
                    
                    # ✅ CHỈ DÙNG COLOR (không Face, không Distance)
                    combined = c_s
                    
                    if combined > best_score: 
                        best_score, best_id, temp_box = combined, tid, box
                
                if best_id is not None and best_score > 0.35:  # ✅ Ngưỡng thấp
                    log_error("Success", f"F{frame_count}: ✅ TÌM LẠI ID {best_id} (Color: {best_score:.2f})", "INFO")
                    current_target_id = best_id
                    found_box = temp_box
                    lost_counter = 0  # ✅ TÌM LẠI ĐƯỢC → RESET
                else:
                    tracking_history.append((frame_count, pred[0][0], pred[1][0], False))
                    continue
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 🔍 MODE NORMAL (CHẠY KHI lost_counter ≤ 60 HOẶC frame % 30 == 1)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif frame_count % 30 == 1:  # ✅ QUÉT KỸ MỖI 1 GIÂY
                best_score, best_id, temp_box = 0, None, None
                for box, tid in zip(all_b, all_ids):
                    f_s = 0
                    x1, y1, x2, y2 = box.astype(int)
                    f_crop = frame[max(0,y1):min(height, y1+int((y2-y1)*0.5)), max(0,x1):min(width,x2)]
                    
                    blur_val = is_blur(f_crop, 90)
                    
                    # ✅ CHẠY FACE RE-ID NẾU KHÔNG MỜ + KHÔNG PHẢI CPU
                    if not blur_val and ref_embedding is not None and DEVICE_STR != "cpu":
                        emb = extract_embedding(enhance_face(f_crop))
                        if emb is not None: f_s = cosine_similarity(ref_embedding, emb)
                    
                    c_s = compare_colors(ref_hist, get_color_histogram(frame, box)) if ref_hist is not None else 0
                    
                    # ✅ BỎ TRỌNG SỐ DISTANCE (vì người có thể ở xa)
                    if not blur_val and f_s > 0:
                        combined = (f_s * 0.6) + (c_s * 0.4)  # Face 60%, Color 40%
                    else:
                        combined = c_s  # Chỉ dùng màu nếu mờ
                    
                    if combined > best_score: 
                        best_score, best_id, temp_box = combined, tid, box

                if best_id is not None and best_score > 0.5:  # ✅ Ngưỡng thấp hơn (0.5 thay vì 0.65)
                    if current_target_id != best_id:
                        log_error("Success", f"F{frame_count}: Chốt ID {best_id} (Score: {best_score:.2f})", "INFO")
                        current_target_id = best_id
                    found_box = temp_box
                    lost_counter = 0  # ✅ TÌM THẤY → RESET

        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 3: LƯU KẾT QUẢ VÀ CẬP NHẬT KALMAN
        # ═══════════════════════════════════════════════════════════════
        if found_box is not None:
            cx, cy = (found_box[0]+found_box[2])/2, (found_box[1]+found_box[3])/2
            kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            tracking_history.append((frame_count, cx, cy, True))
            prev_cx, prev_cy = cx, cy
        else:
            tracking_history.append((frame_count, pred[0][0], pred[1][0], False))
        
        # Giải phóng bộ nhớ định kỳ
        if frame_count % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # --- 5. PASS 2: RENDER VIDEO (DYNAMIC EMA & SMART SMOOTHING) ---
    log_error("Render", f"Bắt đầu xử lý Pass 2: Nội suy & Smooth quỹ đạo (Dynamic Alpha)...", "INFO")
    
    # 1. Nội suy các vị trí bị mất dấu (Gaps)
    # max_gap=fps*0.5 nghĩa là nếu mất dấu dưới 0.5 giây, máy tự nối điểm cũ và mới
    centers = linear_interpolate_center(tracking_history, max_gap=int(fps*0.5))
    
    # 2. Áp dụng Dynamic EMA để camera không bị 'đuổi hình bắt bóng'
    smoothed_centers = [centers[0]]
    prev_smooth = centers[0]
    
    for i in range(1, len(centers)):
        curr_raw = centers[i]
        
        # Tính khoảng cách dịch chuyển (tốc độ tức thời của Hanbin)
        dist = np.sqrt((curr_raw[0] - prev_smooth[0])**2 + (curr_raw[1] - prev_smooth[1])**2)
        
        # Công thức Dynamic Alpha:
        # Nếu đứng yên: alpha = 0.001 (siêu mượt)
        # Nếu di chuyển: alpha tăng dần theo quãng đường (max 0.1 để bám kịp zoom 2x)
        dynamic_alpha = np.clip(0.001 + (dist / width) * 0.8, 0.001, 0.1)
        
        # Tính vị trí mới dựa trên trọng số biến thiên
        new_center = curr_raw * dynamic_alpha + prev_smooth * (1 - dynamic_alpha)
        smoothed_centers.append(new_center)
        prev_smooth = new_center

    centers = np.array(smoothed_centers)
    
    # Thiết lập thông số Crop cho Zoom 2x
    fancam_h, fancam_w = height, int(height * 9 / 16)
    crop_h, crop_w = height / zoom_level, (height / zoom_level) * (9 / 16)
    
    log_error("Render", f"Đang ghi tệp video tạm thời: {output_path_temp}", "INFO")
    
    out_v = cv2.VideoWriter(output_path_temp, cv2.VideoWriter_fourcc(*"XVID"), fps, (fancam_w, fancam_h))
    cap_v = cv2.VideoCapture(video_path)
    
    for f_idx in range(total_frames):
        ret, f_orig = cap_v.read()
        if not ret: break
        
        # Lấy tọa độ tâm đã được làm mượt
        c_idx = min(f_idx, len(centers) - 1)
        cx, cy = centers[c_idx]
        
        # Tính toán tọa độ góc trái trên (Top-Left) để Crop
        l = int(np.clip(cx - crop_w/2, 0, width - int(crop_w)))
        t = int(np.clip(cy - crop_h/2, 0, height - int(crop_h)))
        
        # Thực hiện Crop và Resize về 9:16 dọc
        crop = f_orig[t:t+int(crop_h), l:l+int(crop_w)]
        if crop.size != 0:
            crop_res = cv2.resize(crop, (fancam_w, fancam_h), interpolation=cv2.INTER_LANCZOS4)
            out_v.write(crop_res)
        
        # Giải phóng bộ nhớ frame cũ ngay lập tức
        del f_orig
        if f_idx % 100 == 0:
            gc.collect()

    cap_v.release()
    out_v.release()

    # --- 6. AUDIO SYNC & FINAL PACKAGING ---
    log_error("Audio", "Đang tiến hành đồng bộ âm thanh gốc và nén video chuẩn H.264...", "INFO")
    
    f_out = os.path.basename(video_path).rsplit('.', 1)[0] + f"_fancam_final.mp4"
    
    try:
        # Load video câm và audio gốc
        v_clip = VideoFileClip(output_path_temp)
        a_clip = AudioFileClip(video_path)
        
        # Gán audio và ép thời lượng (duration) bằng nhau để tránh lệch tiếng
        final_clip = v_clip.set_duration(a_clip.duration).set_audio(a_clip)
        
        # Xuất file cuối cùng
        final_clip.write_videofile(
            f_out, 
            codec='libx264', 
            audio_codec='aac', 
            fps=fps, 
            preset='medium', # Cân bằng giữa tốc độ và dung lượng cho máy yếu
            logger=None
        )
        
        # Dọn dẹp tệp tạm
        v_clip.close(); a_clip.close()
        if os.path.exists(output_path_temp):
            os.remove(output_path_temp)
            
        log_error("Success", f"Fancam đã hoàn thành rực rỡ! Đường dẫn: {f_out}", "INFO")
        return "Tạo Fancam thành công!", os.path.abspath(f_out)
        
    except Exception as e:
        log_error("Audio", f"Lỗi trong bước đóng gói cuối cùng: {str(e)}", "ERROR")
        if os.path.exists(output_path_temp):
            return f"Video đã tạo (không tiếng) tại: {output_path_temp}", os.path.abspath(output_path_temp)
        return "Lỗi đồng bộ âm thanh.", None
    