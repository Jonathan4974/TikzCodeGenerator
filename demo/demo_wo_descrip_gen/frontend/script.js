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
let allResults = [];
let currentBrowserIndex = -1;
let selectedThumbIndex = -1;

// ===== Utility =====
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// ===== FIX 1: Global drag prevention =====
document.addEventListener('dragover', preventDefaults, false);
document.addEventListener('drop', preventDefaults, false);

// ===== Drop zone events =====
['dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
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
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;
        imageInput.dispatchEvent(new Event('change'));
    }
}

// ===== FIX 2: Image preview – force removal of inline display styles =====
imageInput.addEventListener('change', () => {
    if (imageInput.files.length === 0) {
        // Hide preview, show hint
        imagePreview.classList.add('hidden');
        imagePreview.style.display = '';       // clear inline
        dropHint.classList.remove('hidden');
        dropHint.style.display = '';          // clear inline
        imagePreview.src = '';
        return;
    }
    const file = imageInput.files[0];
    try {
        const url = URL.createObjectURL(file);
        imagePreview.src = url;
        // Force display block and remove hidden class
        imagePreview.style.display = 'block';
        imagePreview.classList.remove('hidden');
        dropHint.classList.add('hidden');
        dropHint.style.display = 'none';
        // Also handle loading errors
        imagePreview.onload = () => {
            imagePreview.style.display = 'block';
            imagePreview.classList.remove('hidden');
            dropHint.classList.add('hidden');
            dropHint.style.display = 'none';
        };
        imagePreview.onerror = () => {
            // Fallback to FileReader
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                imagePreview.classList.remove('hidden');
                dropHint.classList.add('hidden');
                dropHint.style.display = 'none';
            };
            reader.readAsDataURL(file);
        };
        // Safety timeout in case onload never fires (cached image)
        setTimeout(() => {
            if (imagePreview.src && imagePreview.src !== '') {
                imagePreview.style.display = 'block';
                imagePreview.classList.remove('hidden');
                dropHint.classList.add('hidden');
                dropHint.style.display = 'none';
            }
        }, 200);
    } catch (err) {
        console.error('Preview failed:', err);
        // FileReader fallback
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            imagePreview.classList.remove('hidden');
            dropHint.classList.add('hidden');
            dropHint.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

// ===== Navigation buttons =====
function updateNavButtons(index) {
    if (allResults.length === 0) {
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }
    prevBtn.disabled = (index <= 0);
    nextBtn.disabled = (index >= allResults.length - 1);
}

// ===== FIX 3: Display result – always show either image or error =====
function displayBrowserResult(index) {
    if (index < 0 || index >= allResults.length) {
        resultImage.classList.add('hidden');
        resultImage.style.display = '';
        errorMessage.classList.add('hidden');
        errorMessage.style.display = '';
        loadingPlaceholder.classList.remove('hidden');
        loadingPlaceholder.style.display = 'flex';
        currentIndexSpan.textContent = '0';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        document.getElementById("detailDescription").classList.add("hidden");
        return;
    }

    const data = allResults[index];
    console.log('data:', data);
    console.log('description:', data.description);
    currentIndexSpan.textContent = index + 1;
    totalSamplesSpan.textContent = allResults.length;

    updateNavButtons(index);

    // Hide loading
    loadingPlaceholder.classList.add('hidden');
    loadingPlaceholder.style.display = 'none';

    const descEl = document.getElementById("detailDescription");

    if (data.success && data.png) {
        // Show image
        resultImage.classList.remove('hidden');
        resultImage.style.display = 'block';
        resultImage.src = data.png + '?t=' + new Date().getTime();
        errorMessage.classList.add('hidden');
        errorMessage.style.display = 'none';

        if (data.description && data.description.trim()) {
            descEl.textContent = data.description;
            descEl.classList.remove("hidden");
            // descEl.style.display = "block";
        } else {
            descEl.classList.add("hidden");
            // descEl.style.display = "none";
        }
    } else {
        // Show error message (full content as string)
        resultImage.classList.add('hidden');
        resultImage.style.display = 'none';
        errorMessage.classList.remove('hidden');
        errorMessage.style.display = 'block';
        // FIX: display full error details (including any extra info)
        errorMessage.textContent = data.error || 'Generation or compilation failed.';
        if (data.raw) {
            errorMessage.textContent += '\n\n' + data.raw;
        }
        descEl.classList.add("hidden");
        descEl.style.display = "none";
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

    // Click sync: update main preview and code
    item.addEventListener('click', () => {
        document.querySelectorAll('.thumbnail-item').forEach(el => el.classList.remove('selected'));
        item.classList.add('selected');
        selectedThumbIndex = index;

        currentBrowserIndex = index;
        displayBrowserResult(index);

        const result = allResults[index];
        tikzCode.textContent = (result && result.tikz) ? result.tikz : 'No TikZ code available.';
    });

    thumbnailList.appendChild(item);
    thumbCountSpan.textContent = thumbnailList.children.length;
}

// ===== Handle SSE result =====
function handleResult(data) {
    const idx = allResults.length;
    allResults.push(data);

    totalSamplesSpan.textContent = allResults.length;

    // Auto-display first result
    if (allResults.length === 1) {
        currentBrowserIndex = 0;
        displayBrowserResult(0);
    }

    if (data.success && data.png) {
        addThumbnail(idx, data.png);
    }

    if (currentBrowserIndex !== -1) {
        updateNavButtons(currentBrowserIndex);
    }
}

// ===== Reset UI =====
function resetUI() {
    allResults = [];
    currentBrowserIndex = -1;
    selectedThumbIndex = -1;
    thumbnailList.innerHTML = '';
    thumbCountSpan.textContent = '0';
    totalSamplesSpan.textContent = '0';
    currentIndexSpan.textContent = '0';
    resultImage.classList.add('hidden');
    resultImage.style.display = '';
    document.getElementById("detailDescription").classList.add("hidden");
    document.getElementById("detailDescription").style.display = "none";
    errorMessage.classList.add('hidden');
    errorMessage.style.display = '';
    loadingPlaceholder.classList.remove('hidden');
    loadingPlaceholder.style.display = 'flex';
    loadingPlaceholder.textContent = 'Waiting for generation...';
    tikzCode.textContent = 'Waiting for generation...';
    prevBtn.disabled = true;
    nextBtn.disabled = true;
}

// ===== Generate =====
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
                    // done
                }
            }
        }

        // After stream ends
        if (allResults.length === 0) {
            loadingPlaceholder.classList.remove('hidden');
            loadingPlaceholder.style.display = 'flex';
            loadingPlaceholder.textContent = 'No results returned.';
            prevBtn.disabled = true;
            nextBtn.disabled = true;
        } else {
            // Ensure first result is shown if not already
            if (currentBrowserIndex === -1) {
                currentBrowserIndex = 0;
                displayBrowserResult(0);
            } else {
                displayBrowserResult(currentBrowserIndex);
            }
            updateNavButtons(currentBrowserIndex);

            // Auto-select first thumbnail if exists
            const firstThumb = thumbnailList.querySelector('.thumbnail-item');
            if (firstThumb) {
                firstThumb.click();
            } else {
                // No successful thumbnails, but at least show first error
                if (allResults.length > 0 && currentBrowserIndex === -1) {
                    currentBrowserIndex = 0;
                    displayBrowserResult(0);
                }
            }
        }

    } catch (err) {
        alert('Error: ' + err.message);
        resetUI();
        loadingPlaceholder.classList.remove('hidden');
        loadingPlaceholder.style.display = 'flex';
        loadingPlaceholder.textContent = 'An error occurred. Please try again.';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
    } finally {
        generateButton.disabled = false;
        generateButton.innerText = 'Generate TikZ';
    }
});

// ===== Navigation =====
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
        const result = allResults[currentBrowserIndex];
        tikzCode.textContent = (result && result.tikz) ? result.tikz : 'No TikZ code available.';
    }
});

nextBtn.addEventListener('click', () => {
    if (currentBrowserIndex < allResults.length - 1) {
        currentBrowserIndex++;
        displayBrowserResult(currentBrowserIndex);
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
        const result = allResults[currentBrowserIndex];
        tikzCode.textContent = (result && result.tikz) ? result.tikz : 'No TikZ code available.';
    }
});

// ===== Copy =====
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