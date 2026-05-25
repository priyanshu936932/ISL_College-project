// ISL Recognition - Main App Logic
console.log('✅ App.js loading...');

function waitForDOM() {
    return new Promise((resolve) => {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', resolve);
        } else {
            resolve();
        }
    });
}

waitForDOM().then(() => {
    console.log('✅ DOM Ready');
    initializeApp();
});

function initializeApp() {
    // DOM Elements
    const videoUpload = document.getElementById('video-upload');
    const uploadAnalyzeBtn = document.getElementById('upload-analyze-btn');
    const dropZone = document.getElementById('drop-zone');
    const videoPreview = document.getElementById('video-preview');
    const fileInfo = document.getElementById('file-info');
    const framesInfo = document.getElementById('frames-info');
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    const predictionIcon = document.getElementById('prediction-icon');
    const predictionName = document.getElementById('prediction-name');
    const confidenceValue = document.getElementById('confidence-value');
    const classProbs = document.getElementById('class-probabilities');
    const probContainer = document.getElementById('prob-bars-container');
    const hitCounter = document.getElementById('hit-counter');
    const changeVideoBtn = document.getElementById('change-video-btn');
    const videoSource = document.getElementById('video-source');

    if (!videoUpload || !uploadAnalyzeBtn || !dropZone) {
        console.error('❌ Required DOM elements not found!');
        return;
    }

    // Video error handling
    videoPreview.addEventListener('error', (e) => {
        console.error('❌ Video error:', e);
        console.error('Video error code:', videoPreview.error?.code);
        showToast('❌ Video format not supported. Try MP4 with H.264 codec');
    });

    videoPreview.addEventListener('loadstart', () => {
        console.log('🎬 Video loading started...');
    });

    videoPreview.addEventListener('loadeddata', () => {
        console.log('✅ Video loaded successfully!');
    });

    // Sign Icons Mapping
    const signIcons = {
        'alive': '❤️',
        'bad': '👎',
        'female': '👩',
        'good': '👍',
        'happy': '😊',
        'long': '🙌',
        'male': '👨',
        'default': '👋'
    };

    // Initialize hit counter from localStorage
    let hitCount = parseInt(localStorage.getItem('isl_hit_count') || '0');
    hitCounter.textContent = hitCount;

    console.log('📌 Setting up event handlers...');

    // ==================== DRAG & DROP ====================
    dropZone.addEventListener('click', (e) => {
        e.stopPropagation();
        videoUpload.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            videoUpload.files = dataTransfer.files;
            handleVideoFileSelect(files[0]);
        }
    });

    // File input change
    videoUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleVideoFileSelect(file);
        }
    });

    // ==================== FILE SELECTION ====================
    function handleVideoFileSelect(file) {
        console.log('🎬 File selected:', file.name);

        if (!file.type.startsWith('video/')) {
            showToast('❌ Please select a valid video file (MP4, AVI, MOV)');
            uploadAnalyzeBtn.disabled = true;
            return;
        }

        const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
        if (sizeInMB > 100) {
            showToast('❌ File too large (max 100MB)');
            uploadAnalyzeBtn.disabled = true;
            return;
        }

        // Show video preview
        const url = URL.createObjectURL(file);
        videoPreview.style.display = 'block';
        videoSource.src = url;
        videoSource.type = 'video/mp4';
        videoPreview.load();  // Load the video

        // Hide drop zone content and show file info
        const dropZoneContent = dropZone.querySelector('.drop-zone-content');
        if (dropZoneContent) {
            dropZoneContent.style.display = 'none';
        }

        fileInfo.innerHTML = `✅ Selected: <strong>${file.name}</strong> (${sizeInMB}MB)`;
        fileInfo.style.display = 'block';

        uploadAnalyzeBtn.disabled = false;
        changeVideoBtn.style.display = 'block';
        showToast(`✅ Video ready: ${file.name}`);
    }

    // ==================== ANALYZE VIDEO ====================
    uploadAnalyzeBtn.addEventListener('click', async () => {
        if (!videoUpload.files || videoUpload.files.length === 0) {
            showToast('⚠️ Please select a video file first');
            return;
        }

        const file = videoUpload.files[0];
        uploadAnalyzeBtn.disabled = true;
        uploadAnalyzeBtn.textContent = '⏳ Processing...';

        predictionName.textContent = 'Processing...';
        predictionIcon.textContent = '⏳';
        confidenceValue.textContent = '-';

        const formData = new FormData();
        formData.append('video', file);

        try {
            console.log('📤 Sending to backend...');
            const response = await fetch('http://localhost:5000/predict-video', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            console.log('✅ Prediction result:', result);

            // Update prediction display
            const gestureName = result.prediction.toUpperCase();
            const confidence = result.confidence;
            const confidencePercent = (confidence * 100).toFixed(1);

            predictionName.textContent = gestureName;
            predictionIcon.textContent = signIcons[result.prediction.toLowerCase()] || '👋';
            confidenceValue.textContent = confidencePercent + '%';

            // Update confidence circle background
            const angle = (confidence * 360).toFixed(0);
            document.querySelector('.confidence-circle').style.background = 
                `conic-gradient(var(--purple) 0deg, var(--purple) ${angle}deg, var(--bg-secondary) ${angle}deg)`;

            // Show all class probabilities
            if (result.all_predictions && Object.keys(result.all_predictions).length > 0) {
                displayClassProbabilities(result.all_predictions);
            }

            // Update frames info
            framesInfo.innerHTML = `📹 Frames extracted: <strong>${result.frames_extracted}</strong>`;
            framesInfo.style.display = 'block';

            // Increment hit counter
            hitCount++;
            hitCounter.textContent = hitCount;
            localStorage.setItem('isl_hit_count', hitCount);

            // Highlight matching gesture card
            document.querySelectorAll('.gesture-card').forEach(card => {
                card.classList.remove('active');
                if (card.dataset.gesture === result.prediction.toLowerCase()) {
                    card.classList.add('active');
                }
            });

            showToast(`✅ Prediction: ${gestureName} (${confidencePercent}%)`);

        } catch (error) {
            console.error('❌ Error:', error);
            predictionName.textContent = 'Error';
            predictionIcon.textContent = '❌';
            confidenceValue.textContent = '-';
            showToast('❌ Error: ' + error.message);
        } finally {
            uploadAnalyzeBtn.disabled = false;
            uploadAnalyzeBtn.textContent = '🔍 Analyze Video';
        }
    });

    // ==================== DISPLAY CLASS PROBABILITIES ====================
    function displayClassProbabilities(allPreds) {
        probContainer.innerHTML = '';
        classProbs.style.display = 'block';

        // Sort by probability descending
        const sorted = Object.entries(allPreds).sort((a, b) => b[1] - a[1]);

        sorted.forEach(([className, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.style.marginBottom = '15px';

            const label = document.createElement('div');
            label.className = 'class-prob-label';
            label.innerHTML = `
                <span class="class-prob-label-name">${className.charAt(0).toUpperCase() + className.slice(1)}</span>
                <span class="class-prob-label-value">${percentage}%</span>
            `;

            const bar = document.createElement('div');
            bar.className = 'class-prob-bar';
            const fill = document.createElement('div');
            fill.className = 'class-prob-fill';
            fill.style.width = percentage + '%';
            bar.appendChild(fill);

            item.appendChild(label);
            item.appendChild(bar);
            probContainer.appendChild(item);
        });
    }

    // ==================== TOAST ==================== 
    function showToast(message) {
        toastMessage.textContent = message;
        toast.classList.remove('hidden');
        
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }

    // ==================== INITIALIZATION ====================
    console.log('✅ ISL Recognition App initialized');
    console.log('🎬 Ready for video uploads');
    
    // Set active nav link
    setActiveNavLink();
}

// ==================== NAVIGATION ====================
function setActiveNavLink() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage || (currentPage === '' && href === 'index.html')) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Run on page load
setActiveNavLink();
