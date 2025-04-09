document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const labelUpload = document.getElementById('labelUpload'); // Added
    const uploadedImage = document.getElementById('uploadedImage');
    const canvasContainer = document.getElementById('canvasContainer');
    const promptCanvas = document.getElementById('promptCanvas');
    const ctx = promptCanvas.getContext('2d');
    const promptTypeRadios = document.querySelectorAll('input[name="promptType"]');
    const textPromptSection = document.getElementById('textPromptSection');
    const textPromptInput = document.getElementById('textPrompt');
    const segmentBtn = document.getElementById('segmentBtn');
    const clearBtn = document.getElementById('clearBtn');
    const statusEl = document.getElementById('status');
    const labelOutputArea = document.getElementById('labelOutputArea'); // Added
    const outputLabelImg = document.getElementById('outputLabel');
    const outputMaskImg = document.getElementById('outputMask');

    // --- State Variables ---
    let currentPromptType = 'points';
    let originalImageDataURL = null;
    let originalImageWidth = 0;
    let originalImageHeight = 0;
    let displayedImageWidth = 0;
    let displayedImageHeight = 0;
    let uploadedLabelDataURL = null; // Added - Store if label was uploaded

    let point = null;
    let boundingBox = null;
    let isDrawing = false;
    let startX, startY;

    // --- Canvas Setup & Helpers ---
    function resizeCanvasToImage() {
        if (uploadedImage.naturalWidth && uploadedImage.style.display !== 'none') {
            const rect = uploadedImage.getBoundingClientRect();
            displayedImageWidth = rect.width;
            displayedImageHeight = rect.height;
            promptCanvas.width = displayedImageWidth;
            promptCanvas.height = displayedImageHeight;
            promptCanvas.style.position = 'absolute';
            promptCanvas.style.left = uploadedImage.offsetLeft + 'px';
            promptCanvas.style.top = uploadedImage.offsetTop + 'px';
            promptCanvas.style.pointerEvents = 'auto';
            originalImageWidth = uploadedImage.naturalWidth;
            originalImageHeight = uploadedImage.naturalHeight;
            clearCanvas();
            resetPromptData(false); // Reset prompts but keep label state on resize
            console.log(`Canvas resized to displayed: ${displayedImageWidth}x${displayedImageHeight}, Original: ${originalImageWidth}x${originalImageHeight}`);
        } else {
             promptCanvas.width = 0;
             promptCanvas.height = 0;
             promptCanvas.style.pointerEvents = 'none';
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, promptCanvas.width, promptCanvas.height);
    }

     function resetPromptData(clearLabel = true) { // Added flag
        point = null;
        boundingBox = null;
        textPromptInput.value = "";
        if (clearLabel) {
             uploadedLabelDataURL = null; // Clear label state only if requested
             labelUpload.value = null;    // Clear the file input field
             labelOutputArea.style.display = 'none'; // Hide label output area
        }
    }

    // ... (keep drawing functions: drawPoint, drawBoundingBox, drawScribbleSegment) ...
    function drawPoint(x, y) { ctx.fillStyle = 'red'; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); }
    function drawBoundingBox(x, y, width, height) { ctx.strokeStyle = 'lime'; ctx.lineWidth = 2; ctx.strokeRect(x, y, width, height); }
    function drawScribbleSegment(x1, y1, x2, y2) { ctx.strokeStyle = 'red'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

    // ... (keep scaling functions: scaleCoordsToOriginal, scaleBBoxToOriginal) ...
    function scaleCoordsToOriginal(x, y) { if (!displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) return { x: 0, y: 0 }; const scaleX = originalImageWidth / displayedImageWidth; const scaleY = originalImageHeight / displayedImageHeight; return { x: x * scaleX, y: y * scaleY }; }
    function scaleBBoxToOriginal(bbox) { if (!bbox || !displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) return null; const scaleX = originalImageWidth / displayedImageWidth; const scaleY = originalImageHeight / displayedImageHeight; return { x: bbox.x * scaleX, y: bbox.y * scaleY, width: bbox.width * scaleX, height: bbox.height * scaleY }; }

    // --- Event Handlers ---

    // Main Image Upload
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImageDataURL = e.target.result;
                uploadedImage.src = originalImageDataURL;
                uploadedImage.style.display = 'block';
                outputLabelImg.src = '#'; // Clear previous results
                outputMaskImg.src = '#';
                outputLabelImg.style.display = 'none'; // Hide results areas
                outputMaskImg.style.display = 'none';
                labelOutputArea.style.display = 'none'; // Hide label output area specifically

                // Clear associated label state when new main image is uploaded
                resetPromptData(true); // Reset everything including label

                uploadedImage.onload = () => {
                     resizeCanvasToImage();
                     statusEl.textContent = 'Status: Image loaded. Optionally upload label. Select prompt type and interact.';
                };
                 uploadedImage.onerror = () => {
                     statusEl.textContent = 'Status: Error loading image.';
                     originalImageDataURL = null;
                }
            }
            reader.readAsDataURL(file);
        }
    });

    // Label Image Upload (Optional) - Added
    labelUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
             const reader = new FileReader();
             reader.onload = (e) => {
                 uploadedLabelDataURL = e.target.result; // Store label data URL
                 console.log("Optional label uploaded.");
                 // No visual change needed here, just store the state
                 // Could optionally display a thumbnail if desired
             }
             reader.readAsDataURL(file);
        } else {
             uploadedLabelDataURL = null; // Clear state if file selection is cancelled
        }
    });

    window.addEventListener('resize', resizeCanvasToImage);

    // Prompt Type Selection
    promptTypeRadios.forEach(radio => {
        radio.addEventListener('change', (event) => {
            currentPromptType = event.target.value;
            textPromptSection.style.display = currentPromptType === 'text' ? 'block' : 'none';
            clearCanvas();
            resetPromptData(false); // Don't clear uploaded label state when changing prompt type
            statusEl.textContent = `Status: Switched to ${currentPromptType} prompt.`;
            promptCanvas.style.cursor = (currentPromptType === 'points') ? 'crosshair' : /* ... other cursors ... */ 'default';
        });
    });

    // ... (Keep canvas mousedown, mousemove, mouseup, mouseout listeners exactly as in the previous 'single-point' version) ...
    promptCanvas.addEventListener('mousedown', (event) => { if (!originalImageDataURL) return; const rect = promptCanvas.getBoundingClientRect(); startX = event.clientX - rect.left; startY = event.clientY - rect.top; isDrawing = true; if (currentPromptType === 'points') { clearCanvas(); const scaledPoint = scaleCoordsToOriginal(startX, startY); point = scaledPoint; drawPoint(startX, startY); console.log('Point set (original coords):', point); isDrawing = false; } else if (currentPromptType === 'scribble') { /* Start scribble */ } else if (currentPromptType === 'bbox') { clearCanvas(); boundingBox = { x: startX, y: startY, width: 0, height: 0 }; console.log('BBox start:', startX, startY); } });
    promptCanvas.addEventListener('mousemove', (event) => { if (!isDrawing || !originalImageDataURL) return; const rect = promptCanvas.getBoundingClientRect(); const currentX = event.clientX - rect.left; const currentY = event.clientY - rect.top; if (currentPromptType === 'scribble') { drawScribbleSegment(startX, startY, currentX, currentY); startX = currentX; startY = currentY; } else if (currentPromptType === 'bbox') { clearCanvas(); const width = currentX - startX; const height = currentY - startY; drawBoundingBox(startX, startY, width, height); } });
    promptCanvas.addEventListener('mouseup', (event) => { if (!isDrawing || !originalImageDataURL) return; isDrawing = false; const rect = promptCanvas.getBoundingClientRect(); const endX = event.clientX - rect.left; const endY = event.clientY - rect.top; if (currentPromptType === 'bbox') { const finalX = Math.min(startX, endX); const finalY = Math.min(startY, endY); const finalWidth = Math.abs(endX - startX); const finalHeight = Math.abs(endY - startY); if (finalWidth > 1 && finalHeight > 1) { clearCanvas(); boundingBox = { x: finalX, y: finalY, width: finalWidth, height: finalHeight }; drawBoundingBox(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height); console.log('BBox finalized (displayed coords):', boundingBox); } else { boundingBox = null; clearCanvas(); console.log('BBox canceled (too small).'); } } else if (currentPromptType === 'scribble') { console.log('Scribble finished.'); } });
    promptCanvas.addEventListener('mouseout', () => { if (isDrawing) { isDrawing = false; /* Handle cancelling drag if needed */ } });


    // Clear Button
    clearBtn.addEventListener('click', () => {
        clearCanvas();
        resetPromptData(true); // Reset everything including label
        statusEl.textContent = 'Status: Prompts cleared.';
    });

    // Segment Button
    segmentBtn.addEventListener('click', async () => {
        if (!originalImageDataURL) {
            statusEl.textContent = 'Status: Error - Please upload an image first.';
            return;
        }
        // ... (keep prompt data preparation logic exactly as before, checking for point, bbox etc.) ...
        let promptDataToSend = null;
        let finalPromptType = currentPromptType;
        if (finalPromptType === 'points') { if (!point) { statusEl.textContent = 'Status: Error - Please click a point...'; return; } promptDataToSend = [point]; }
        else if (finalPromptType === 'bbox') { if (!boundingBox || boundingBox.width <= 0 || boundingBox.height <= 0) { statusEl.textContent = 'Status: Error - Please draw a valid box...'; return; } promptDataToSend = scaleBBoxToOriginal(boundingBox); if (!promptDataToSend) { statusEl.textContent = 'Status: Error - Could not scale box...'; return; } }
        else if (finalPromptType === 'scribble') { try { promptDataToSend = promptCanvas.toDataURL('image/png'); } catch (e) { statusEl.textContent = 'Status: Error - Could not get scribble...'; console.error("Canvas toDataURL error:", e); return; } }
        else if (finalPromptType === 'text') { promptDataToSend = textPromptInput.value.trim(); if (!promptDataToSend) { statusEl.textContent = 'Status: Error - Please enter text...'; return; } }
        else { statusEl.textContent = 'Status: Error - Invalid prompt type...'; return; }


        statusEl.textContent = `Status: Sending ${finalPromptType} prompt to backend...`;
        // Hide previous results before sending request
        outputLabelImg.style.display = 'none';
        outputMaskImg.style.display = 'none';
        labelOutputArea.style.display = 'none'; // Ensure label area is hidden initially

        // --- Send data to backend ---
        try {
            const response = await fetch('/segment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: originalImageDataURL,
                    prompt_type: finalPromptType,
                    prompt_data: promptDataToSend,
                    original_width: originalImageWidth,
                    original_height: originalImageHeight
                }),
            });

            const result = await response.json();

            if (response.ok) {
                statusEl.textContent = `Status: ${result.message || 'Success!'}`;

                // --- Modified Result Display Logic ---
                // Display Label Output *only if* a label was originally uploaded
                if (uploadedLabelDataURL && result.output_label_b64) {
                    outputLabelImg.src = result.output_label_b64;
                    labelOutputArea.style.display = 'block'; // Show the label area
                    outputLabelImg.style.display = 'block'; // Ensure image inside is visible
                } else {
                    labelOutputArea.style.display = 'none'; // Keep label area hidden
                }

                // Display Mask Output (always show if available)
                if (result.output_mask_b64) {
                    outputMaskImg.src = result.output_mask_b64;
                    outputMaskImg.style.display = 'block';
                }
                // --- End Modified Result Display Logic ---

            } else {
                statusEl.textContent = `Status: Error - ${result.error || response.statusText}`;
                 console.error("Backend Error:", result.error);
            }

        } catch (error) {
            statusEl.textContent = 'Status: Error - Could not connect to backend.';
            console.error('Fetch Error:', error);
        }
    });

    // Initial setup
    promptCanvas.style.pointerEvents = 'none';
    statusEl.textContent = 'Status: Waiting for image upload.';
    labelOutputArea.style.display = 'none'; // Ensure label output is hidden initially

});