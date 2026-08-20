const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const description = document.getElementById("description");
const generateButton = document.getElementById("generateButton");
const sampleCountSelect = document.getElementById("sampleCount");
const resultImage = document.getElementById("resultImage");
const errorMessage = document.getElementById("errorMessage");
const loadingPlaceholder = document.getElementById("loadingPlaceholder");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const currentIndexSpan = document.getElementById("currentIndex");
const totalSamplesSpan = document.getElementById("totalSamples");
const thumbnailList = document.getElementById("thumbnailList");
const thumbCountSpan = document.getElementById("thumbCount");
const tikzCode = document.getElementById("tikzCode");
const copyBtn = document.getElementById("copyBtn");
const dropZone = document.getElementById('dropZone');
const dropHint = document.getElementById('dropHint');

// State
let allResults = [];          // Array of result objects
let currentBrowserIndex = -1; // Index in allResults currently shown
let selectedThumbIndex = -1;  // Index in allResults of the selected thumbnail

// ===== Utility: prevent default behavior =====
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// ===== 1. Fix drag-and-drop: prevent globally on document =====
document.addEventListener('dragover', preventDefaults, false);
document.addEventListener('drop', preventDefaults, false);

// ===== dropZone drag events =====
['dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop zone on dragover
dropZone.addEventListener('dragover', () => {
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

// Handle drop
dropZone.addEventListener('drop', handleDrop);

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');

    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please drop an image file.');
            return;
        }
        // Update the files property of the file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;
        // Manually trigger change event for preview
        imageInput.dispatchEvent(new Event('change'));
    }
}

// ===== 2. Fix image preview: use classList to toggle visibility =====
imageInput.addEventListener('change', () => {
    if (imageInput.files.length === 0) {
        imagePreview.classList.add('hidden');
        dropHint.classList.remove('hidden');
        imagePreview.src = '';
        return;
    }
    const file = imageInput.files[0];
    // Use FileReader as fallback, more reliable
    try {
        const url = URL.createObjectURL(file);
        imagePreview.src = url;
        imagePreview.onload = () => {
            imagePreview.classList.remove('hidden');
            dropHint.classList.add('hidden');
        };
        imagePreview.onerror = () => {
            // If createObjectURL fails, try FileReader
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                dropHint.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        };
        // If image is cached, onload may not fire, so show directly
        // Safety: delayed display
        setTimeout(() => {
            if (imagePreview.src && imagePreview.src !== '') {
                imagePreview.classList.remove('hidden');
                dropHint.classList.add('hidden');
            }
        }, 100);
    } catch (err) {
        console.error('Preview failed:', err);
        // Fallback to FileReader
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropHint.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
});

// ===== Update navigation buttons =====
function updateNavButtons(index) {
    if (allResults.length === 0) {
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }
    prevBtn.disabled = (index <= 0);
    nextBtn.disabled = (index >= allResults.length - 1);
}

// ===== Display the result in the browser =====
function displayBrowserResult(index) {
    if (index < 0 || index >= allResults.length) {
        resultImage.classList.add('hidden');
        errorMessage.classList.add('hidden');
        loadingPlaceholder.classList.remove('hidden');
        currentIndexSpan.textContent = '0';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }

    const data = allResults[index];
    currentIndexSpan.textContent = index + 1;
    totalSamplesSpan.textContent = allResults.length;

    updateNavButtons(index);

    loadingPlaceholder.classList.add('hidden');

    if (data.success && data.png) {
        resultImage.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        // Add timestamp to prevent caching
        resultImage.src = data.png + '?t=' + new Date().getTime();
    } else {
        resultImage.classList.add('hidden');
        errorMessage.classList.remove('hidden');
        errorMessage.textContent = data.error || 'Generation or compilation failed.';
    }
}

// ===== Add thumbnail =====
function addThumbnail(index, pngUrl) {
    const item = document.createElement('div');
    item.className = 'thumbnail-item';
    item.dataset.index = index;

    const img = document.createElement('img');
    img.src = pngUrl + '?t=' + new Date().getTime();
    img.alt = `Sample ${index+1}`;
    img.loading = 'lazy';
    item.appendChild(img);

    // ===== 6. Thumbnail click sync: update large preview + code =====
    item.addEventListener('click', () => {
        document.querySelectorAll('.thumbnail-item').forEach(el => el.classList.remove('selected'));
        item.classList.add('selected');
        selectedThumbIndex = index;

        currentBrowserIndex = index;
        displayBrowserResult(index);

        const result = allResults[index];
        if (result && result.tikz) {
            tikzCode.textContent = result.tikz;
        } else {
            tikzCode.textContent = 'No TikZ code available.';
        }
    });

    thumbnailList.appendChild(item);
    thumbCountSpan.textContent = thumbnailList.children.length;
}

// ===== Handle each result from SSE stream =====
function handleResult(data) {
    const idx = allResults.length;
    allResults.push(data);

    totalSamplesSpan.textContent = allResults.length;

    // If first result, display it
    if (allResults.length === 1) {
        currentBrowserIndex = 0;
        displayBrowserResult(0);
    }

    // If successful and has png, add thumbnail
    if (data.success && data.png) {
        addThumbnail(idx, data.png);
    }

    // Update nav buttons state
    if (currentBrowserIndex !== -1) {
        updateNavButtons(currentBrowserIndex);
    }
}

// ===== Reset all UI state =====
function resetUI() {
    allResults = [];
    currentBrowserIndex = -1;
    selectedThumbIndex = -1;
    thumbnailList.innerHTML = '';
    thumbCountSpan.textContent = '0';
    totalSamplesSpan.textContent = '0';
    currentIndexSpan.textContent = '0';
    resultImage.classList.add('hidden');
    errorMessage.classList.add('hidden');
    loadingPlaceholder.classList.remove('hidden');
    loadingPlaceholder.textContent = 'Waiting for generation...';
    tikzCode.textContent = 'Waiting for generation...';
    prevBtn.disabled = true;
    nextBtn.disabled = true;
}

// ===== Generate button click handler =====
generateButton.addEventListener('click', async () => {
    if (imageInput.files.length === 0) {
        alert('Please upload an image first.');
        return;
    }

    generateButton.disabled = true;
    generateButton.innerText = 'Generating...';
    resetUI();

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('description', description.value);
    formData.append('num_samples', sampleCountSelect.value);

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === '') continue;
                    try {
                        const data = JSON.parse(dataStr);
                        handleResult(data);
                    } catch (e) {
                        console.error('Failed to parse JSON:', dataStr, e);
                    }
                } else if (line.startsWith('event: done')) {
                    // Generation finished
                }
            }
        }

        // ===== After stream ends =====
        if (allResults.length === 0) {
            loadingPlaceholder.classList.remove('hidden');
            loadingPlaceholder.textContent = 'No results returned.';
            prevBtn.disabled = true;
            nextBtn.disabled = true;
        } else {
            // Ensure browser index is valid
            if (currentBrowserIndex === -1) {
                currentBrowserIndex = 0;
                displayBrowserResult(0);
            } else {
                // Refresh current display
                displayBrowserResult(currentBrowserIndex);
            }

            // ===== 4. Force refresh button states =====
            updateNavButtons(currentBrowserIndex);

            // Auto-select first thumbnail
            const firstThumb = thumbnailList.querySelector('.thumbnail-item');
            if (firstThumb) {
                firstThumb.click();
            } else {
                // No successful results, show first error
                if (allResults.length > 0) {
                    currentBrowserIndex = 0;
                    displayBrowserResult(0);
                }
            }
        }

    } catch (err) {
        alert('Error: ' + err.message);
        resetUI();
        loadingPlaceholder.classList.remove('hidden');
        loadingPlaceholder.textContent = 'An error occurred. Please try again.';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
    } finally {
        generateButton.disabled = false;
        generateButton.innerText = 'Generate TikZ';
    }
});

// ===== Navigation button events =====
prevBtn.addEventListener('click', () => {
    if (currentBrowserIndex > 0) {
        currentBrowserIndex--;
        displayBrowserResult(currentBrowserIndex);
        // Sync thumbnail highlight
        const items = thumbnailList.querySelectorAll('.thumbnail-item');
        items.forEach((el, idx) => {
            const dataIdx = parseInt(el.dataset.index);
            if (dataIdx === currentBrowserIndex) {
                el.classList.add('selected');
                selectedThumbIndex = currentBrowserIndex;
            } else {
                el.classList.remove('selected');
            }
        });
        // Sync code display
        const result = allResults[currentBrowserIndex];
        if (result && result.tikz) {
            tikzCode.textContent = result.tikz;
        }
    }
});

nextBtn.addEventListener('click', () => {
    if (currentBrowserIndex < allResults.length - 1) {
        currentBrowserIndex++;
        displayBrowserResult(currentBrowserIndex);
        // Sync thumbnail highlight
        const items = thumbnailList.querySelectorAll('.thumbnail-item');
        items.forEach((el, idx) => {
            const dataIdx = parseInt(el.dataset.index);
            if (dataIdx === currentBrowserIndex) {
                el.classList.add('selected');
                selectedThumbIndex = currentBrowserIndex;
            } else {
                el.classList.remove('selected');
            }
        });
        // Sync code display
        const result = allResults[currentBrowserIndex];
        if (result && result.tikz) {
            tikzCode.textContent = result.tikz;
        }
    }
});

// ===== Copy button =====
copyBtn.addEventListener('click', () => {
    const code = tikzCode.textContent;
    if (code && code !== 'Waiting for generation...' && code !== 'No TikZ code available.') {
        navigator.clipboard.writeText(code).then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => copyBtn.textContent = 'Copy', 2000);
        }).catch(err => {
            alert('Copy failed: ' + err);
        });
    } else {
        alert('No TikZ code to copy');
    }
});

// ===== Initial state =====
resetUI();