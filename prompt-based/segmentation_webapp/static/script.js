document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
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
    const outputLabelImg = document.getElementById('outputLabel');
    const outputMaskImg = document.getElementById('outputMask');

    // --- State Variables ---
    let currentPromptType = 'points'; // Default
    let originalImageDataURL = null;
    let originalImageWidth = 0;
    let originalImageHeight = 0;
    let displayedImageWidth = 0;
    let displayedImageHeight = 0;

    // ***** CHANGE 1: Store a single point object or null *****
    let point = null; // Stores the single {x, y} scaled coordinates for 'Points' mode
    // let points = []; // Old way - storing multiple points

    let boundingBox = null; // {x, y, width, height} (scaled coordinates)
    let isDrawing = false;
    let startX, startY;
    // Scribble state is implicitly the canvas content in this version

    // --- Canvas Setup ---
    function resizeCanvasToImage() {
        // ... (keep this function as before) ...
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
            resetPromptData(); // Reset stored prompts
            console.log(`Canvas resized to displayed: ${displayedImageWidth}x${displayedImageHeight}, Original: ${originalImageWidth}x${originalImageHeight}`);
        } else {
             promptCanvas.width = 0;
             promptCanvas.height = 0;
             promptCanvas.style.pointerEvents = 'none';
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, promptCanvas.width, promptCanvas.height);
        // No need to clear scribblePaths if not storing them extensively
    }

     function resetPromptData() {
        // ***** CHANGE 2: Reset single point state *****
        point = null;
        // points = []; // Old way

        boundingBox = null;
        textPromptInput.value = "";
    }

    // --- Drawing Functions ---
    function drawPoint(x, y) {
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
    }

    // ... (keep drawBoundingBox, drawScribbleSegment as before) ...
    function drawBoundingBox(x, y, width, height) {
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
    }
    function drawScribbleSegment(x1, y1, x2, y2) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }


    // ... (keep scaleCoordsToOriginal, scaleBBoxToOriginal as before) ...
     function scaleCoordsToOriginal(x, y) {
        if (!displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) {
            return { x: 0, y: 0 };
        }
        const scaleX = originalImageWidth / displayedImageWidth;
        const scaleY = originalImageHeight / displayedImageHeight;
        return {
            x: x * scaleX,
            y: y * scaleY
        };
    }
    function scaleBBoxToOriginal(bbox) {
        if (!bbox || !displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) {
            return null;
        }
        const scaleX = originalImageWidth / displayedImageWidth;
        const scaleY = originalImageHeight / displayedImageHeight;
        return {
            x: bbox.x * scaleX,
            y: bbox.y * scaleY,
            width: bbox.width * scaleX,
            height: bbox.height * scaleY
        };
    }


    // --- Event Handlers ---

    // ... (keep imageUpload listener as before) ...
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImageDataURL = e.target.result;
                uploadedImage.src = originalImageDataURL;
                uploadedImage.style.display = 'block';
                outputLabelImg.style.display = 'none';
                outputMaskImg.style.display = 'none';
                uploadedImage.onload = () => {
                     resizeCanvasToImage();
                     statusEl.textContent = 'Status: Image loaded. Select prompt type and interact.';
                };
                 uploadedImage.onerror = () => {
                     statusEl.textContent = 'Status: Error loading image.';
                     originalImageDataURL = null;
                }
            }
            reader.readAsDataURL(file);
        }
    });
    window.addEventListener('resize', resizeCanvasToImage);

    // ... (keep promptTypeRadios listener as before) ...
     promptTypeRadios.forEach(radio => {
        radio.addEventListener('change', (event) => {
            currentPromptType = event.target.value;
            textPromptSection.style.display = currentPromptType === 'text' ? 'block' : 'none';
            clearCanvas(); // Clear drawings when changing type
            resetPromptData(); // Also reset stored data
            statusEl.textContent = `Status: Switched to ${currentPromptType} prompt.`;
            promptCanvas.style.cursor = (currentPromptType === 'points') ? 'crosshair' :
                                         (currentPromptType === 'bbox') ? 'crosshair' :
                                         (currentPromptType === 'scribble') ? 'crosshair' :
                                         'default';
        });
    });

    // Canvas Interaction (Points, Bbox, Scribble)
    promptCanvas.addEventListener('mousedown', (event) => {
        if (!originalImageDataURL) return;

        const rect = promptCanvas.getBoundingClientRect();
        startX = event.clientX - rect.left;
        startY = event.clientY - rect.top;
        isDrawing = true;

        // ***** CHANGE 3: Handle single point logic *****
        if (currentPromptType === 'points') {
            clearCanvas(); // Clear previous point visual
            const scaledPoint = scaleCoordsToOriginal(startX, startY);
            point = scaledPoint; // Update the single point state
            drawPoint(startX, startY); // Draw the new point visual
            console.log('Point set (original coords):', point);
            isDrawing = false; // Points are single clicks, no dragging needed
        }
        // ***** End CHANGE 3 *****
        else if (currentPromptType === 'scribble') {
             // Start drawing scribble
        } else if (currentPromptType === 'bbox') {
             // Start drawing bbox
             // Ensure canvas is clear before starting new box visual
             clearCanvas();
             boundingBox = { x: startX, y: startY, width: 0, height: 0 }; // Store displayed coords initially
             console.log('BBox start:', startX, startY);
        }
    });

    // ... (keep mousemove listener mostly as before, just ensure bbox redraw is correct) ...
     promptCanvas.addEventListener('mousemove', (event) => {
        if (!isDrawing || !originalImageDataURL) return;

        const rect = promptCanvas.getBoundingClientRect();
        const currentX = event.clientX - rect.left;
        const currentY = event.clientY - rect.top;

        if (currentPromptType === 'scribble') {
            drawScribbleSegment(startX, startY, currentX, currentY);
            startX = currentX;
            startY = currentY;
        } else if (currentPromptType === 'bbox') {
            clearCanvas(); // Clear previous temporary box
            const width = currentX - startX; // Use startX from mousedown
            const height = currentY - startY; // Use startY from mousedown
            drawBoundingBox(startX, startY, width, height); // Draw current box outline
        }
    });


    // ... (keep mouseup listener as before) ...
     promptCanvas.addEventListener('mouseup', (event) => {
        if (!isDrawing || !originalImageDataURL) return;
        isDrawing = false;

        const rect = promptCanvas.getBoundingClientRect();
        const endX = event.clientX - rect.left;
        const endY = event.clientY - rect.top;

        if (currentPromptType === 'bbox') {
            // Use startX/startY from mousedown stored in the closure
            const finalX = Math.min(startX, endX);
            const finalY = Math.min(startY, endY);
            const finalWidth = Math.abs(endX - startX);
            const finalHeight = Math.abs(endY - startY);

             if (finalWidth > 1 && finalHeight > 1) { // Require a minimum size
                 clearCanvas(); // Clear temporary drawing
                 boundingBox = { x: finalX, y: finalY, width: finalWidth, height: finalHeight };
                 drawBoundingBox(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
                 console.log('BBox finalized (displayed coords):', boundingBox);
             } else {
                 boundingBox = null;
                 clearCanvas();
                 console.log('BBox canceled (too small).');
             }
        } else if (currentPromptType === 'scribble') {
             console.log('Scribble finished.');
        }
        // No specific action needed for 'points' on mouseup
    });

    // ... (keep mouseout listener as before) ...
      promptCanvas.addEventListener('mouseout', () => {
           if (isDrawing) { // Stop drawing if mouse leaves while dragging
                isDrawing = false;
                if (currentPromptType === 'bbox') {
                    // Optional: Cancel the bbox if mouse leaves? Or finalize?
                    // Let's finalize based on last known position for simplicity,
                    // or you could cancel like this:
                    // boundingBox = null;
                    // clearCanvas();
                    // console.log('BBox canceled (mouse out).');
                } else if (currentPromptType === 'scribble') {
                     console.log('Scribble segment ended (mouse out).');
                }
           }
      });

    // Clear Button
    clearBtn.addEventListener('click', () => {
        clearCanvas();
        resetPromptData(); // Resets point, bbox, text
        statusEl.textContent = 'Status: Prompts cleared.';
    });

    // Segment Button
    segmentBtn.addEventListener('click', async () => {
        if (!originalImageDataURL) {
            statusEl.textContent = 'Status: Error - Please upload an image first.';
            return;
        }

        let promptDataToSend = null;
        let finalPromptType = currentPromptType;

        // ***** CHANGE 4: Prepare single point data *****
        if (finalPromptType === 'points') {
            if (!point) { // Check if the single point state is set
                statusEl.textContent = 'Status: Error - Please click a point for point prompt.';
                return;
            }
            // Backend expects a list of points, so wrap the single point in an array
            promptDataToSend = [point];
        }
        // ***** End CHANGE 4 *****
        else if (finalPromptType === 'bbox') {
             if (!boundingBox || boundingBox.width <= 0 || boundingBox.height <= 0) {
                 statusEl.textContent = 'Status: Error - Please draw a valid bounding box.';
                 return;
             }
             promptDataToSend = scaleBBoxToOriginal(boundingBox);
             if (!promptDataToSend) { // Check if scaling failed
                 statusEl.textContent = 'Status: Error - Could not scale bounding box.';
                 return;
             }
        } else if (finalPromptType === 'scribble') {
            try {
                // Add check for empty canvas? Difficult, let backend handle empty mask.
                promptDataToSend = promptCanvas.toDataURL('image/png');
            } catch (e) {
                statusEl.textContent = 'Status: Error - Could not get scribble data from canvas.';
                console.error("Canvas toDataURL error:", e);
                return;
            }
        } else if (finalPromptType === 'text') {
            promptDataToSend = textPromptInput.value.trim();
            if (!promptDataToSend) {
                statusEl.textContent = 'Status: Error - Please enter text for text prompt.';
                return;
            }
        } else {
            statusEl.textContent = 'Status: Error - Invalid prompt type selected.';
            return;
        }

        statusEl.textContent = `Status: Sending ${finalPromptType} prompt to backend...`;
        outputLabelImg.style.display = 'none';
        outputMaskImg.style.display = 'none';

        // --- Send data to backend ---
        try {
            const response = await fetch('/segment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: originalImageDataURL,
                    prompt_type: finalPromptType,
                    prompt_data: promptDataToSend, // This will be [point] for 'points' mode
                    original_width: originalImageWidth,
                    original_height: originalImageHeight
                }),
            });
            // ... (keep response handling as before) ...
             const result = await response.json();
            if (response.ok) {
                statusEl.textContent = `Status: ${result.message || 'Success!'}`;
                if (result.output_label_b64) {
                    outputLabelImg.src = result.output_label_b64;
                    outputLabelImg.style.display = 'block';
                }
                 if (result.output_mask_b64) {
                    outputMaskImg.src = result.output_mask_b64;
                    outputMaskImg.style.display = 'block';
                }
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

});