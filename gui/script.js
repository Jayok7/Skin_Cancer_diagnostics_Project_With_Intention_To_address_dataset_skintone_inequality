/**
 * Skin Cancer Classification Frontend
 * Handles image upload, server communication, and results display
 */

// DOM elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');

// State
let currentFile = null;

// ============================================
// UPLOAD HANDLING
// ============================================

// Click to upload
uploadZone.addEventListener('click', () => {
    if (!imagePreview.style.display || imagePreview.style.display === 'none') {
        fileInput.click();
    }
});

// File selection
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
});

// Remove image
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

/**
 * Handle file upload and validation
 */
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;

        // Hide results if showing
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Reset upload state
 */
function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadPlaceholder.style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

// ============================================
// ANALYSIS
// ============================================

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show loading state
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    analyzeBtn.disabled = true;

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('image', currentFile);

        // ADD THIS LINE: Send the chosen model to the server
        const selectedModel = document.getElementById('modelSelect').value;
        formData.append('model_choice', selectedModel);

        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        // Reset button state
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

// ============================================
// RESULTS DISPLAY
// ============================================

/**
 * Display prediction results
 */
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    // Update prediction
    const predictionValue = document.getElementById('predictionValue');
    predictionValue.textContent = data.predicted_name;

    // Update confidence
    const confidence = (data.confidence * 100).toFixed(1);
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');

    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = confidence + '%';
    }, 100);

    confidenceText.textContent = `Confidence: ${confidence}%`;

    // Update images
    document.getElementById('originalImage').src = data.original_image;
    document.getElementById('gradcamImage').src = data.gradcam_image || data.original_image;

    // Update all predictions
    displayAllPredictions(data.all_confidences);

    // VLM Output
    const vlmOutputElement = document.getElementById('vlmOutput');
    if (data.vlm_report) {
        // Translate Markdown to HTML
        let formattedText = data.vlm_report
            // 1. Find anything between ** and **, and wrap it in HTML bold tags
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // 2. Find numbers followed by a dot and space (e.g., "1. ", "2. ") and add line breaks before them
            .replace(/(\s|^)(\d+\.\s)/g, '<br><br>$2')
            // 3. Add line breaks before "Conclusion:" (whether it got bolded in step 1 or not)
            .replace(/(<strong>Conclusion:<\/strong>|Conclusion:)/gi, '<br><br>$1');

        // Clean up any accidental double breaks at the very beginning of the text
        formattedText = formattedText.replace(/^(<br>)+/, '');

        // Use innerHTML so the browser renders the bold tags and line breaks
        vlmOutputElement.innerHTML = formattedText;
    } else {
        vlmOutputElement.textContent = "VLM reasoning could not be generated. Please check server logs.";
    }
}

/**
 * Display all classification scores
 */
function displayAllPredictions(confidences) {
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';

    // Sort by confidence descending
    const sorted = Object.entries(confidences).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([name, confidence]) => {
        const percentage = (confidence * 100).toFixed(1);

        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-name">${name}</div>
            <div class="prediction-bar-container">
                <div class="prediction-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="prediction-percentage">${percentage}%</div>
        `;

        predictionsList.appendChild(item);
    });
}

// ============================================
// ERROR HANDLING
// ============================================

/**
 * Show error message
 */
function showError(message) {
    alert(message);
}

// ============================================
// INITIALIZATION
// ============================================

// Check server health on load and populate models
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        // Populate the dropdown menu!
        const modelSelect = document.getElementById('modelSelect');
        if (data.available_models && data.available_models.length > 0) {
            modelSelect.innerHTML = ''; // Clear loading text
            data.available_models.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName;
                modelSelect.appendChild(option);
            });
            console.log(`✓ Server ready. Loaded ${data.available_models.length} models.`);
        } else {
            modelSelect.innerHTML = '<option value="">No models found on server</option>';
            console.warn('Warning: No CNN models loaded on server');
        }

    } catch (error) {
        console.error('Server health check failed:', error);
    }
});

// Auto-hide header on scroll
(() => {
    const header = document.querySelector('header');
    let lastScrollY = window.scrollY;

    window.addEventListener('scroll', () => {
        const currentScrollY = window.scrollY;

        if (currentScrollY > lastScrollY && currentScrollY > 80) {
            // Scrolling DOWN & past 80px — hide header
            header.classList.add('header-hidden');
        } else {
            // Scrolling UP — show header
            header.classList.remove('header-hidden');
        }

        lastScrollY = currentScrollY;
    });
})();

// ============================================
// THEME TOGGLE
// ============================================
const themeToggle = document.getElementById('themeToggle');

// Check for saved theme preference or system preference
const savedTheme = localStorage.getItem('theme');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
    document.documentElement.setAttribute('data-theme', 'dark');
}

// Toggle theme on click
themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
});