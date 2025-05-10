document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const labelUpload = document.getElementById('labelUpload'); // Added
    const uploadedImage = document.getElementById('uploadedImage');
    const canvasContainer = document.getElementById('canvasContainer');
    const promptCanvas = document.getElementById('promptCanvas');
    const ctx = promptCanvas.getContext('2d');
    const modelTypeRadios = document.querySelectorAll('input[name="modelType"]'); // Changed selector name
    // const textPromptSection = document.getElementById('textPromptSection'); // Removed - No longer needed
    // const textPromptInput = document.getElementById('textPrompt'); // Removed - No longer needed
    const segmentBtn = document.getElementById('segmentBtn');
    const clearBtn = document.getElementById('clearBtn');
    const statusEl = document.getElementById('status');
    // Removed duplicate declaration of statusEl
    // Image/Output Column Elements (New Structure)
    const originalImageColumn = document.getElementById('originalImageColumn'); // Added for reference if needed
    const predictedMaskColumn = document.getElementById('predictedMaskColumn');
    const outputMaskImg = document.getElementById('outputMask');
    const groundTruthColumn = document.getElementById('groundTruthColumn');
    const outputGroundTruth = document.getElementById('outputGroundTruth');
    const legendContainer = document.getElementById('legendContainer');
    // Removed outputContainer, labelOutputArea, outputLabelImg references

    // --- State Variables ---
    let currentModelType = 'unet'; // Changed variable name and default
    let originalImageDataURL = null;
    let originalImageWidth = 0;
    let originalImageHeight = 0;
    let displayedImageWidth = 0;
    let displayedImageHeight = 0;
    let uploadedLabelDataURL = null; // Added - Store if label was uploaded

    let point = null;
    // let boundingBox = null; // Removed - BBox not used anymore
    let isDrawing = false; // Still needed for points
    let startX, startY; // Still needed for points
    // --- Color Map (for Legend) ---
    const COLOR_MAP_JS = {
        0: 'rgb(0, 0, 0)',        // Black
        1: 'rgb(255, 0, 0)',      // Red
        2: 'rgb(0, 255, 0)',      // Green
        3: 'rgb(0, 0, 255)'       // Blue
        // Add more if needed, ensure consistency with backend COLOR_MAP keys
    };


    // --- Canvas Setup & Helpers ---
    function resizeCanvasToImage() {
        if (uploadedImage.naturalWidth && uploadedImage.style.display !== 'none') {
            const rect = uploadedImage.getBoundingClientRect();
            displayedImageWidth = rect.width;
            displayedImageHeight = rect.height;
            promptCanvas.width = displayedImageWidth;
            promptCanvas.height = displayedImageHeight;
            // Ensure canvas is positioned correctly over the image within its container
            promptCanvas.style.position = 'absolute';
            // Get position relative to the container, not the whole page
            promptCanvas.style.left = uploadedImage.offsetLeft + 'px';
            promptCanvas.style.top = uploadedImage.offsetTop + 'px';
            promptCanvas.style.pointerEvents = 'auto'; // Enable interaction
            originalImageWidth = uploadedImage.naturalWidth;
            originalImageHeight = uploadedImage.naturalHeight;
            clearCanvas();
            resetPromptData(false); // Reset prompts but keep label state on resize
            // Re-apply interaction state based on current model
            const selectedModelRadio = document.querySelector('input[name="modelType"]:checked');
            updateCanvasInteractivity(selectedModelRadio);
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

     function resetPromptData(clearLabel = true) {
        point = null; // Keep point reset
        // boundingBox = null; // Removed
        // textPromptInput.value = ""; // Removed
        if (clearLabel) {
             uploadedLabelDataURL = null; // Clear label state only if requested
             labelUpload.value = null;    // Clear the file input field
            // Clear the ground truth image if label is cleared, but keep column visible
            // groundTruthColumn.style.display = 'none'; // Keep column visible
            outputGroundTruth.src = '#';
            outputGroundTruth.style.display = 'none'; // Hide only the image
            // No need to manage grid columns via JS anymore
            // Removed labelOutputArea logic
         }
    }

    // --- Drawing Functions ---
    // Only need drawPoint now
    function drawPoint(x, y) { ctx.fillStyle = 'red'; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); }
    // function drawBoundingBox(x, y, width, height) { ctx.strokeStyle = 'lime'; ctx.lineWidth = 2; ctx.strokeRect(x, y, width, height); } // Removed
    // function drawScribbleSegment(x1, y1, x2, y2) { ctx.strokeStyle = 'red'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); } // Removed

    // ... (keep scaling functions: scaleCoordsToOriginal, scaleBBoxToOriginal) ...
    // Only need scaleCoordsToOriginal now
    function scaleCoordsToOriginal(x, y) { if (!displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) return { x: 0, y: 0 }; const scaleX = originalImageWidth / displayedImageWidth; const scaleY = originalImageHeight / displayedImageHeight; return { x: x * scaleX, y: y * scaleY }; }
    // function scaleBBoxToOriginal(bbox) { if (!bbox || !displayedImageWidth || !displayedImageHeight || !originalImageWidth || !originalImageHeight) return null; const scaleX = originalImageWidth / displayedImageWidth; const scaleY = originalImageHeight / displayedImageHeight; return { x: bbox.x * scaleX, y: bbox.y * scaleY, width: bbox.width * scaleX, height: bbox.height * scaleY }; } // Removed

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
                // Clear output images, keep columns visible
                outputMaskImg.src = '#';
                outputMaskImg.style.display = 'none'; // Hide image
                // predictedMaskColumn.style.display = 'none'; // Keep column visible

                outputGroundTruth.src = '#';
                outputGroundTruth.style.display = 'none'; // Hide image
                // groundTruthColumn.style.display = 'none'; // Keep column visible

                legendContainer.style.display = 'none'; // Hide legend
                legendContainer.innerHTML = '<h4>Legend:</h4>'; // Clear legend content
                // Removed labelOutputArea, outputLabelImg, outputContainer logic

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

    // Label Image Upload (Optional) - Reverted
    labelUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
             const reader = new FileReader();
             reader.onload = (e) => {
                 uploadedLabelDataURL = e.target.result; // Store label data URL
                 console.log("Optional label uploaded.");
                 // No visual change needed here in reverted version
             }
             reader.readAsDataURL(file);
        } else {
             uploadedLabelDataURL = null; // Clear state if file selection is cancelled
        }
    });

    window.addEventListener('resize', resizeCanvasToImage);

    // --- Helper to Enable/Disable Canvas ---
    function updateCanvasInteractivity(selectedRadio) {
        const requiresPrompt = selectedRadio.dataset.requiresPrompt === 'true';
        if (requiresPrompt) {
            promptCanvas.style.pointerEvents = 'auto';
            promptCanvas.style.cursor = 'crosshair';
            statusEl.textContent = `Status: Model '${currentModelType}' selected. Click a point on the image.`;
        } else {
            promptCanvas.style.pointerEvents = 'none';
            promptCanvas.style.cursor = 'default';
            statusEl.textContent = `Status: Model '${currentModelType}' selected. No prompt needed. Click Segment.`;
        }
    }

    // Model Type Selection
    modelTypeRadios.forEach(radio => {
        radio.addEventListener('change', (event) => {
            currentModelType = event.target.value;
            clearCanvas();
            resetPromptData(false); // Reset point data, keep label
            updateCanvasInteractivity(event.target); // Update canvas based on selection
        });
    });
    // Initial check on page load
    const initiallySelectedModelRadio = document.querySelector('input[name="modelType"]:checked');
    if (initiallySelectedModelRadio) {
        currentModelType = initiallySelectedModelRadio.value;
        updateCanvasInteractivity(initiallySelectedModelRadio);
    }

    // ... (Keep canvas mousedown, mousemove, mouseup, mouseout listeners exactly as in the previous 'single-point' version) ...
    // Canvas Interaction (Only for Points when Prompt Model is selected)
    promptCanvas.addEventListener('mousedown', (event) => {
        if (!originalImageDataURL || currentModelType !== 'prompt_model') return; // Only allow if prompt model selected
        const rect = promptCanvas.getBoundingClientRect();
        startX = event.clientX - rect.left;
        startY = event.clientY - rect.top;
        isDrawing = true; // Keep isDrawing flag? Maybe not needed if only one click

        // Handle point click directly
        clearCanvas();
        const scaledPoint = scaleCoordsToOriginal(startX, startY);
        point = scaledPoint; // Store the single point
        drawPoint(startX, startY); // Draw feedback on canvas
        console.log('Point set (original coords):', point);
        isDrawing = false; // Point is set on mousedown
    });

    // Remove mousemove, mouseup, mouseout listeners as they were for bbox/scribble
    // promptCanvas.addEventListener('mousemove', ...);
    // promptCanvas.addEventListener('mouseup', ...);
    // promptCanvas.addEventListener('mouseout', ...);


    // Clear Button
    clearBtn.addEventListener('click', () => {
        clearCanvas();
       resetPromptData(true); // Reset everything including label state
       // Clear output images, keep columns visible
       outputMaskImg.src = '#';
       outputMaskImg.style.display = 'none'; // Hide image
       // predictedMaskColumn.style.display = 'none'; // Keep column visible

       outputGroundTruth.src = '#';
       outputGroundTruth.style.display = 'none'; // Hide image
       // groundTruthColumn.style.display = 'none'; // Keep column visible

       legendContainer.style.display = 'none'; // Hide legend
       legendContainer.innerHTML = '<h4>Legend:</h4>'; // Clear legend content
       // Removed labelOutputArea, outputLabelImg, outputContainer logic
        statusEl.textContent = 'Status: Prompts and outputs cleared.';
    });

    // Segment Button
    segmentBtn.addEventListener('click', async () => {
        if (!originalImageDataURL) {
            statusEl.textContent = 'Status: Error - Please upload an image first.';
            return;
        }
        // --- Prepare Payload ---
        const payload = {
            image_b64: originalImageDataURL,
            model_type: currentModelType, // Send selected model type
            original_width: originalImageWidth,
            original_height: originalImageHeight,
            label_b64: uploadedLabelDataURL // Send label if available
        };

        // Add prompt data ONLY if the prompt model is selected
        if (currentModelType === 'prompt_model') {
            if (!point) {
                statusEl.textContent = 'Status: Error - Please click a point for the Prompt Model.';
                return;
            }
            payload.prompt_type = 'points'; // Backend expects this field for prompt model
            payload.prompt_data = [point]; // Send the single point in an array
        }


        statusEl.textContent = `Status: Sending request for model '${currentModelType}' to backend...`;
        // Clear output images before sending request, keep columns visible
        outputMaskImg.src = '#';
        outputMaskImg.style.display = 'none';
        outputGroundTruth.src = '#';
        outputGroundTruth.style.display = 'none';
        // predictedMaskColumn.style.display = 'none'; // Keep column visible
        // groundTruthColumn.style.display = 'none'; // Keep column visible
        legendContainer.style.display = 'none'; // Hide legend before request
        legendContainer.innerHTML = '<h4>Legend:</h4>'; // Clear legend content
        // Removed labelOutputArea and outputContainer logic

        // --- Send data to backend ---
        try {
            const response = await fetch('/segment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload), // Send the constructed payload
            });

            const result = await response.json();

            if (response.ok) {
                statusEl.textContent = `Status: ${result.message || 'Success!'}`;
                // --- Updated Output Display Logic (3 Columns) ---

                // 1. Predicted Mask (Column 2)
                if (result.output_mask_b64) {
                    outputMaskImg.src = result.output_mask_b64;
                    outputMaskImg.style.display = 'block'; // Show image
                    // predictedMaskColumn.style.display = 'block'; // Column is already visible
                } else {
                    outputMaskImg.src = '#';
                    outputMaskImg.style.display = 'none'; // Hide image
                    // predictedMaskColumn.style.display = 'none'; // Keep column visible
                }

                // 2. Ground Truth Mask (Column 3)
                // Show only if a label was uploaded AND the backend returned the processed label
                if (uploadedLabelDataURL && result.output_label_b64) {
                    outputGroundTruth.src = result.output_label_b64; // Use the processed label from backend
                    outputGroundTruth.style.display = 'block'; // Show image
                    // groundTruthColumn.style.display = 'block'; // Column is already visible
                } else {
                    outputGroundTruth.src = '#';
                    outputGroundTruth.style.display = 'none'; // Hide image
                    // groundTruthColumn.style.display = 'none'; // Keep column visible
                }
                // Removed labelOutputArea and outputContainer logic
                // --- End Updated Output Display Logic ---

                // --- Populate Legend ---
                legendContainer.innerHTML = '<h4>Legend:</h4>'; // Clear previous legend items
                if (result.class_names) {
                    Object.entries(result.class_names).forEach(([index, name]) => {
                        const color = COLOR_MAP_JS[index];
                        if (color) {
                            const legendItem = document.createElement('div');
                            legendItem.classList.add('legend-item'); // Add class for styling
                            legendItem.innerHTML = `
                                <span class="legend-color-box" style="background-color: ${color};"></span>
                                ${name} (Index: ${index})
                            `;
                            legendContainer.appendChild(legendItem);
                        }
                    });
                    legendContainer.style.display = 'block'; // Show legend
                } else {
                    legendContainer.style.display = 'none'; // Hide if no class names received
                }
                // --- End Populate Legend ---

            } else {
                statusEl.textContent = `Status: Error - ${result.error || response.statusText}`;
                 console.error("Backend Error:", result.error);
            }

        } catch (error) {
            statusEl.textContent = 'Status: Error - Could not connect to backend.';
            console.error('Fetch Error:', error);
        }
    });

    // Initial setup is handled by the model type change listener now
    // promptCanvas.style.pointerEvents = 'none'; // Set by listener
    // statusEl.textContent = 'Status: Waiting for image upload.'; // Set by listener
    // Ensure output images (not columns) are hidden initially
    outputMaskImg.style.display = 'none';
    outputGroundTruth.style.display = 'none';
    // Columns are visible by default via CSS
    // Removed labelOutputArea

});