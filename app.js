// ============================================
// NOTIFICATION SYSTEM
// ============================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        warning: '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>',
        error: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>',
        success: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>',
        info: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>'
    };
    
    toast.innerHTML = `
        <div class="toast-icon">
            <svg viewBox="0 0 24 24">${icons[type]}</svg>
        </div>
        <div class="toast-content">${message}</div>
        <button class="toast-close">
            <svg viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        removeToast(toast);
    });
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        removeToast(toast);
    }, 4000);
}

function removeToast(toast) {
    toast.classList.add('closing');
    setTimeout(() => {
        toast.remove();
    }, 300);
}

function showConfirm(title, message, onConfirm, type = 'warning') {
    const dialog = document.getElementById('confirmDialog');
    const titleEl = document.getElementById('confirmTitle');
    const messageEl = document.getElementById('confirmMessage');
    const cancelBtn = document.getElementById('confirmCancel');
    const okBtn = document.getElementById('confirmOk');
    const icon = dialog.querySelector('.confirm-icon');
    
    // Update content
    titleEl.textContent = title;
    messageEl.textContent = message;
    
    // Update icon type
    icon.className = `confirm-icon ${type}`;
    
    // Show dialog
    dialog.style.display = 'flex';
    
    // Handle buttons
    const closeDialog = () => {
        dialog.style.display = 'none';
        cancelBtn.replaceWith(cancelBtn.cloneNode(true));
        okBtn.replaceWith(okBtn.cloneNode(true));
    };
    
    document.getElementById('confirmCancel').addEventListener('click', closeDialog);
    
    document.getElementById('confirmOk').addEventListener('click', () => {
        onConfirm();
        closeDialog();
    });
}

// ============================================
// STATE MANAGEMENT
// ============================================
let videoFile = null;
let faceFiles = [];
let colorFiles = [];
let detectionData = null;

// ============================================
// DOM ELEMENTS
// ============================================
const videoInput = document.getElementById('videoInput');
const dropzone = document.getElementById('dropzone');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const videoPlayer = document.getElementById('videoPlayer');
const detectBtn = document.getElementById('detectBtn');
const targetIdInput = document.getElementById('targetIdInput');
const detectionResult = document.getElementById('detectionResult');
const detectionImage = document.getElementById('detectionImage');
const detectionInfo = document.getElementById('detectionInfo');
const generateBtn = document.getElementById('generateBtn');
const processingModal = document.getElementById('processingModal');
const resultModal = document.getElementById('resultModal');
const progressBar = document.getElementById('progressBar');
const resultVideo = document.getElementById('resultVideo');

// ============================================
// VIDEO UPLOAD
// ============================================
videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        videoFile = file;
        loadVideoPreview(file);
    }
});

// Drag & Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        videoFile = file;
        loadVideoPreview(file);
    }
});

function loadVideoPreview(file) {
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    uploadSection.style.display = 'none';
    previewSection.style.display = 'block';
    console.log('✅ Video loaded:', file.name);
}

// Play button overlay
const playOverlay = document.getElementById('playOverlay');
if (playOverlay) {
    playOverlay.querySelector('.play-btn-large').addEventListener('click', () => {
        videoPlayer.play();
        playOverlay.style.display = 'none';
    });
}

// ============================================
// CLOSE VIDEO BUTTON (XÓA VIDEO + DETECTION)
// ============================================
const closeVideoBtn = document.getElementById('closeVideoBtn');

closeVideoBtn.addEventListener('click', () => {
    showConfirm(
        'Remove Video',
        'Are you sure you want to remove this video? All detection results will be lost.',
        () => {
            // Reset video state
            videoFile = null;
            detectionData = null;
            
            // Stop and clear video player
            videoPlayer.pause();
            videoPlayer.src = '';
            
            // Hide preview and detection result
            previewSection.style.display = 'none';
            detectionResult.style.display = 'none';
            
            // Show upload section again
            uploadSection.style.display = 'block';
            
            // Reset input field
            videoInput.value = '';
            targetIdInput.value = '';
            
            // Clear file previews
            faceFiles = [];
            colorFiles = [];
            document.getElementById('faceFilePreview').innerHTML = '';
            document.getElementById('colorFilePreview').innerHTML = '';
            
            showToast('Video removed successfully', 'success');
            console.log('✅ Video removed successfully');
        },
        'delete'
    );
});

// ============================================
// DETECTION
// ============================================
detectBtn.addEventListener('click', async () => {
    if (!videoFile) {
        showToast('Please upload a video first!', 'warning');
        return;
    }
    
    detectBtn.disabled = true;
    detectBtn.innerHTML = `
        <div class="spinner-ring" style="width:20px;height:20px;border-width:2px;margin:0;display:inline-block"></div>
        <span style="margin-left:8px">Detecting...</span>
    `;
    
    const formData = new FormData();
    formData.append('video', videoFile);
    
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            detectionImage.src = data.image;
            const idList = data.ids.map(id => `ID ${id}`).join(' • ');
            detectionInfo.textContent = `Found ${data.ids.length} person(s): ${idList}`;
            detectionResult.style.display = 'block';
            targetIdInput.placeholder = `Select from: ${data.ids.join(', ')}`;
            
            detectionResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            showToast(`Successfully detected ${data.ids.length} person(s)!`, 'success');
            console.log('✅ Detected:', data.ids);
        } else {
            showToast('Detection failed: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('Detection error:', error);
        showToast('Network error during detection', 'error');
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = `Detect People`;
    }
});

// ============================================
// FACE & COLOR UPLOAD
// ============================================
document.getElementById('faceUpload').addEventListener('change', (e) => {
    faceFiles = Array.from(e.target.files);
    updateFilePreview('faceFilePreview', faceFiles);
});

document.getElementById('colorUpload').addEventListener('change', (e) => {
    colorFiles = Array.from(e.target.files);
    updateFilePreview('colorFilePreview', colorFiles);
});

function updateFilePreview(containerId, files) {
    const container = document.getElementById(containerId);
    container.innerHTML = files.map(f => `
        <div class="file-item">
            <span class="file-name">${f.name}</span>
        </div>
    `).join('');
}

// ============================================
// GENERATE FANCAM
// ============================================
generateBtn.addEventListener('click', async () => {
    const targetId = targetIdInput.value.trim();
    
    if (!videoFile || !targetId) {
        showToast('Please upload video and enter target ID!', 'warning');
        return;
    }
    
    processingModal.style.display = 'flex';
    document.getElementById('processingStatus').textContent = 'Uploading video...';
    progressBar.style.width = '0%';
    
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('target_id', targetId);
    formData.append('zoom_level', document.getElementById('zoomSlider').value);
    
    faceFiles.forEach(f => formData.append('face_images', f));
    colorFiles.forEach(f => formData.append('color_images', f));
    
    // Simulate progress
    let progress = 0;
    const statusTexts = [
        'Analyzing video frames...',
        'Detecting people...',
        'Tracking target person...',
        'Generating fancam video...'
    ];
    let statusIndex = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 3;
        if (progress > 90) progress = 90;
        progressBar.style.width = `${progress}%`;
        
        if (progress > 25 * (statusIndex + 1) && statusIndex < statusTexts.length - 1) {
            statusIndex++;
            document.getElementById('processingStatus').textContent = statusTexts[statusIndex];
        }
    }, 500);
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        clearInterval(interval);
        progressBar.style.width = '100%';
        document.getElementById('processingStatus').textContent = 'Complete!';
        
        setTimeout(() => {
            processingModal.style.display = 'none';
            if (data.success) {
                resultVideo.src = data.video;
                resultModal.style.display = 'flex';
                document.getElementById('downloadBtn').href = data.video;
                showToast('Fancam generated successfully! 🎉', 'success');
            } else {
                showToast('Processing failed: ' + data.message, 'error');
            }
        }, 800);
        
    } catch (error) {
        clearInterval(interval);
        processingModal.style.display = 'none';
        console.error('Processing error:', error);
        showToast('Network error during processing', 'error');
    }
});

// ============================================
// MODAL CONTROLS
// ============================================
document.getElementById('closeResultBtn').addEventListener('click', () => {
    resultModal.style.display = 'none';
});

document.getElementById('createNewBtn').addEventListener('click', () => {
    location.reload();
});

// ============================================
// ZOOM SLIDER & PRESET BUTTONS
// ============================================
const zoomSlider = document.getElementById('zoomSlider');
const zoomValue = document.getElementById('zoomValue');
const presetButtons = document.querySelectorAll('.preset-btn');

// Update zoom value display when slider changes
zoomSlider.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    zoomValue.textContent = `${value.toFixed(1)}x`;
    
    // ✅ Update active state on preset buttons
    updateActivePresetButton(value);
});

// Handle preset button clicks
presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const zoom = parseFloat(btn.dataset.zoom);
        
        // Update slider value
        zoomSlider.value = zoom;
        zoomValue.textContent = `${zoom.toFixed(1)}x`;
        
        // ✅ Update active state
        updateActivePresetButton(zoom);
    });
});

// ✅ Function to update active button state
function updateActivePresetButton(currentZoom) {
    presetButtons.forEach(btn => {
        const btnZoom = parseFloat(btn.dataset.zoom);
        
        // Check if current zoom matches this button's value (with tolerance)
        if (Math.abs(currentZoom - btnZoom) < 0.05) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// ✅ Initialize on page load
updateActivePresetButton(parseFloat(zoomSlider.value));

console.log('✅ Fancam UI initialized');

// ============================================
// RESULT MODAL - CLOSE BUTTON
// ============================================
const closeResultBtn = document.getElementById('closeResultBtn');
const createNewBtn = document.getElementById('createNewBtn');

// ✅ Hàm đóng modal và dừng video hoàn toàn
function closeResultModal() {
    const resultModal = document.getElementById('resultModal');
    const resultVideo = document.getElementById('resultVideo');
    
    // 1. Dừng video
    resultVideo.pause();
    
    // 2. Reset thời gian về 0
    resultVideo.currentTime = 0;
    
    // 3. Xóa nguồn video (giải phóng bộ nhớ)
    resultVideo.src = '';
    resultVideo.load(); // ✅ Quan trọng: Force reload để clear buffer
    
    // 4. Ẩn modal
    resultModal.style.display = 'none';
    
    console.log('✅ Result modal closed, video stopped and memory cleared');
}

// ✅ Nút X (góc trên phải)
closeResultBtn.addEventListener('click', () => {
    closeResultModal();
});

// ✅ Nút "New Project"
createNewBtn.addEventListener('click', () => {
    closeResultModal();
    
    // Reset toàn bộ ứng dụng về trạng thái ban đầu
    videoFile = null;
    detectionData = null;
    
    // Dừng và xóa video preview
    videoPlayer.pause();
    videoPlayer.src = '';
    
    // Ẩn preview và detection
    previewSection.style.display = 'none';
    detectionResult.style.display = 'none';
    
    // Hiển thị upload section
    uploadSection.style.display = 'block';
    
    // Reset inputs
    videoInput.value = '';
    targetIdInput.value = '';
    
    // Xóa reference files
    faceFiles = [];
    colorFiles = [];
    document.getElementById('faceFilePreview').innerHTML = '';
    document.getElementById('colorFilePreview').innerHTML = '';
    
    // Reset zoom
    const zoomSlider = document.getElementById('zoomSlider');
    const zoomValue = document.getElementById('zoomValue');
    zoomSlider.value = 1.0;
    zoomValue.textContent = '1.0x';
    updateActivePresetButton(1.0);
    
    showToast('Ready for new project!', 'success');
    console.log('✅ Application reset to initial state');
});
