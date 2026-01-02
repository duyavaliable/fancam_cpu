from flask import Flask, request, jsonify, send_file, send_from_directory
import os
from main import initial_detection, process_fancam, model, DEVICE_STR
import time

app = Flask(__name__, static_folder='.', static_url_path='')

# ✅ DISABLE CACHING CHO TẤT CẢ STATIC FILES
@app.after_request
def add_no_cache_headers(response):
    if request.path.endswith(('.html', '.css', '.js')):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response

# ✅ ROUTE CHO INDEX.HTML VỚI CACHE BUSTING
@app.route('/')
def index():
    return send_file('fancam_ui.html', cache_timeout=0)

# ✅ STATIC FILES VỚI CACHE BUSTING
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename, cache_timeout=0)

# ============================================
# API: DETECT PEOPLE IN VIDEO
# ============================================
@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No video file uploaded'})
        
        video_file = request.files['video']
        temp_video_path = f"temp_upload_{video_file.filename}"
        video_file.save(temp_video_path)
        
        print(f"🎬 Detecting people in: {temp_video_path}")
        
        # Call initial_detection from main.py
        image_path, message, info = initial_detection(temp_video_path)
        
        if image_path is None:
            return jsonify({
                'success': False,
                'message': message or 'Detection failed'
            })
        
        # Parse IDs from info string
        ids = []
        for line in info.split('\n'):
            if line.startswith('ID '):
                parts = line.split()
                id_num = parts[1]
                ids.append(id_num)
        
        # Convert image to base64
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up temp files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        print(f"✅ Detection complete. Found {len(ids)} people")
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_data}',
            'message': message,
            'info': info,
            'ids': ids
        })
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Detection error: {str(e)}'
        })

# ============================================
# API: PROCESS FANCAM
# ============================================
@app.route('/api/process', methods=['POST'])
def process():
    video_path = None
    face_paths = []
    color_paths = []
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No video file uploaded'})
        
        video_file = request.files['video']
        target_id = request.form.get('target_id', '1')
        zoom_level = float(request.form.get('zoom_level', 1.0))
        
        # Save video temporarily
        video_path = f"temp_process_{video_file.filename}"
        video_file.save(video_path)
        
        print(f"🎬 Processing fancam for ID {target_id} with zoom {zoom_level}x")
        
        # Handle face images
        face_images = request.files.getlist('face_images')
        face_objs = []
        for img in face_images:
            face_path = f"temp_face_{img.filename}"
            img.save(face_path)
            face_paths.append(face_path)
            # Create object with .name attribute like Gradio
            face_obj = type('FileObj', (object,), {'name': face_path})()
            face_objs.append(face_obj)
        
        print(f"📸 Loaded {len(face_objs)} face reference images")
        
        # Handle color images
        color_images = request.files.getlist('color_images')
        color_objs = []
        for img in color_images:
            color_path = f"temp_color_{img.filename}"
            img.save(color_path)
            color_paths.append(color_path)
            color_obj = type('FileObj', (object,), {'name': color_path})()
            color_objs.append(color_obj)
        
        print(f"👕 Loaded {len(color_objs)} outfit reference images")
        
        # Process fancam using main.py function
        result_msg, video_output = process_fancam(
            video_path,
            target_id,
            face_objs,
            color_objs,
            zoom_level
        )
        
        if video_output is None:
            return jsonify({
                'success': False,
                'message': result_msg or 'Processing failed'
            })
        
        print(f"✅ Fancam generated: {video_output}")
        
        # Return video file path
        return jsonify({
            'success': True,
            'message': result_msg,
            'video': f'/output/{os.path.basename(video_output)}'
        })
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return jsonify({
            'success': False,
            'message': f'Processing error: {str(e)}'
        })
    finally:
        # Clean up temporary files
        for path in [video_path] + face_paths + color_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"🗑️ Cleaned up: {path}")
                except:
                    pass

# ============================================
# SERVE OUTPUT VIDEOS
# ============================================
@app.route('/output/<filename>')
def serve_output(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=False, mimetype='video/mp4')
    return jsonify({'error': 'File not found'}), 404

# ============================================
# HEALTH CHECK
# ============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'device': DEVICE_STR,
        'model': 'yolov8n'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 K-POP FANCAM AI SERVER")
    print("="*70)
    print(f"\n📱 Open in browser: http://localhost:5000")
    print(f"⚡ Backend: Flask + YOLOv8 + DeepFace")
    print(f"🎨 Frontend: Modern UI with FlexClip Design")
    print(f"🖥️  Device: {DEVICE_STR.upper()}")
    print(f"\n⚠️  Press Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)