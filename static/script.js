document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const uploadedImage = document.getElementById('uploaded-image');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const imageDescription = document.getElementById('image-description');
    const memeSection = document.getElementById('meme-section');
    const captionTexts = [
        document.getElementById('caption1-text'),
        document.getElementById('caption2-text'),
        document.getElementById('caption3-text')
    ];
    const editCaption = document.getElementById('edit-caption');
    const editButton = document.getElementById('edit-button');
    const generateMemeBtn = document.getElementById('generate-meme');
    const finalMemeSection = document.getElementById('final-meme-section');
    const memeResult = document.getElementById('meme-result');
    const downloadBtn = document.getElementById('download-meme');
    const regenerateBtn = document.getElementById('regenerate-context');
    const regenerateText = document.getElementById('regenerate-text');
    const regeneratingIndicator = document.getElementById('regenerating-indicator');
    let currentFilename = '';
    let selectedCaption = '';
    let isRegenerating = false;

    // Drag and drop functionality
    const dropZone = document.querySelector('.border-dashed');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            imageUpload.files = files;
            handleImageUpload();
        }
    }

    // Handle file input change
    imageUpload.addEventListener('change', handleImageUpload);

    function handleImageUpload() {
        if (!imageUpload.files || !imageUpload.files[0]) return;

        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);

        // Show loading state
        dropZone.classList.add('loading');
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            currentFilename = data.filename;
            
            // Show uploaded image
            uploadedImage.src = `/uploads/${currentFilename}`;
            uploadedImage.classList.remove('hidden');
            uploadedImage.classList.add('show');
            previewPlaceholder.classList.add('hidden');
            
            // Show image description
            imageDescription.textContent = data.description || 'No description available';
            
            // Display captions
            const captions = data.captions || [];
            captionTexts.forEach((element, index) => {
                if (captions[index]) {
                    element.textContent = captions[index];
                    element.parentElement.style.display = 'block';
                } else {
                    element.parentElement.style.display = 'none';
                }
            });
            
            // Set initial selected caption
            selectedCaption = captions[0] || '';
            editCaption.value = selectedCaption;
            
            // Show sections
            memeSection.classList.remove('hidden');
            finalMemeSection.classList.add('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert(error.message || 'Error uploading image');
        })
        .finally(() => {
            dropZone.classList.remove('loading');
        });
    }

    // Handle caption selection
    document.querySelectorAll('input[name="caption"]').forEach((radio, index) => {
        radio.addEventListener('change', async function() {
            if (this.checked) {
                // Update selected caption
                selectedCaption = captionTexts[index].textContent;
                editCaption.value = selectedCaption;
                
                // Generate meme immediately
                await generateMemeWithCaption(selectedCaption);
            }
        });
    });

    // Function to generate meme with a given caption
    async function generateMemeWithCaption(caption) {
        if (!currentFilename || !caption) {
            console.error('Missing filename or caption');
            return;
        }
        
        try {
            // Show loading state
            generateMemeBtn.textContent = 'Generating...';
            generateMemeBtn.classList.add('loading');
            generateMemeBtn.disabled = true;
            
            const response = await fetch('/generate-meme', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: currentFilename,
                    caption: caption
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Show the generated meme
            memeResult.src = data.meme_url + '?t=' + new Date().getTime(); // Add timestamp to prevent caching
            memeResult.onload = function() {
                memeResult.classList.add('show');
                finalMemeSection.classList.remove('hidden');
                // Enable download button
                downloadBtn.disabled = false;
            };
            
        } catch (error) {
            console.error('Error:', error);
            alert(error.message || 'Error generating meme');
        } finally {
            generateMemeBtn.textContent = 'Generate Meme';
            generateMemeBtn.classList.remove('loading');
            generateMemeBtn.disabled = false;
        }
    }

    // Update generate meme button to use the same function
    generateMemeBtn.addEventListener('click', async function() {
        if (!currentFilename || !selectedCaption) {
            alert('Please select an image and caption first');
            return;
        }
        await generateMemeWithCaption(selectedCaption);
    });

    // Handle caption editing
    editButton.addEventListener('click', function() {
        const selectedRadio = document.querySelector('input[name="caption"]:checked');
        if (selectedRadio) {
            const index = selectedRadio.id.replace('caption', '') - 1;
            captionTexts[index].textContent = editCaption.value;
            selectedCaption = editCaption.value;
        }
    });

    regenerateBtn.addEventListener('click', async function() {
        if (!currentFilename || isRegenerating) return;
        
        try {
            // Show loading state
            isRegenerating = true;
            regenerateBtn.disabled = true;
            regenerateBtn.classList.add('loading');
            regenerateText.textContent = 'Regenerating...';
            regeneratingIndicator.classList.remove('hidden');
            
            // Request new description and captions
            const response = await fetch('/regenerate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: currentFilename
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update description
            imageDescription.textContent = data.description;
            
            // Update captions
            const captions = data.captions || [];
            captionTexts.forEach((element, index) => {
                if (captions[index]) {
                    element.textContent = captions[index];
                    element.parentElement.style.display = 'block';
                } else {
                    element.parentElement.style.display = 'none';
                }
            });
            
            // Reset selected caption
            selectedCaption = captions[0] || '';
            editCaption.value = selectedCaption;
            
            // Reset radio buttons
            document.querySelector('input[name="caption"]').checked = true;
            
            // Hide final meme section
            finalMemeSection.classList.add('hidden');
            
        } catch (error) {
            console.error('Error:', error);
            alert(error.message || 'Error regenerating content');
        } finally {
            // Reset button state
            isRegenerating = false;
            regenerateBtn.disabled = false;
            regenerateBtn.classList.remove('loading');
            regenerateText.textContent = 'Regenerate';
            regeneratingIndicator.classList.add('hidden');
        }
    });

    downloadBtn.addEventListener('click', function() {
        if (memeResult.src) {
            const a = document.createElement('a');
            a.href = memeResult.src;
            a.download = 'meme.jpg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    });
});
