
// AIthentic App JavaScript
// This file contains all the JavaScript functionality

// Global variables
let particleAnimationId;

// Utility functions
function scrollToApp() {
    document.getElementById('app').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

function toggleMobileMenu() {
    const navLinks = document.getElementById('navLinks');
    navLinks.classList.toggle('show');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const types = {
        'jpg': 'Image', 'jpeg': 'Image', 'png': 'Image', 'gif': 'Image', 'webp': 'Image',
        'txt': 'Text Document',
        'mp4': 'Video', 'mov': 'Video', 'avi': 'Video', 'mkv': 'Video', 'webm': 'Video'
    };
    return types[ext] || 'Unknown';
}

// File handling
function handleFileSelect(file) {
    const fileInfo = document.getElementById('fileInfo');
    const uploadIcon = document.querySelector('.upload-icon');
    const uploadText = document.querySelector('.upload-text');
    const uploadSubtext = document.querySelector('.upload-subtext');

    // Update UI
    uploadIcon.textContent = '✅';
    uploadText.textContent = 'File ready for analysis';
    uploadSubtext.textContent = 'Click "Analyze Content" to begin';

    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileType').textContent = getFileType(file.name);
    fileInfo.classList.add('show');
}

// Tab functionality
function switchTab(tabType) {
    const fileTab = document.getElementById('fileTab');
    const textTab = document.getElementById('textTab');
    const uploadSection = document.getElementById('uploadSection');
    const textInputSection = document.getElementById('textInputSection');
    const fileInput = document.getElementById('file');
    const textInput = document.getElementById('textContent');

    if (tabType === 'file') {
        fileTab.classList.add('active');
        textTab.classList.remove('active');
        uploadSection.style.display = 'block';
        textInputSection.style.display = 'none';

        // Clear text input and make file input required
        textInput.value = '';
        fileInput.required = true;
        textInput.required = false;
    } else {
        textTab.classList.add('active');
        fileTab.classList.remove('active');
        uploadSection.style.display = 'none';
        textInputSection.style.display = 'block';

        // Clear file input and make text input required
        fileInput.value = '';
        fileInput.required = false;
        textInput.required = true;
        textInput.focus();
    }
}

// Feedback submission
function submitFeedback(feedbackType, sessionId) {
    const feedbackButtons = document.getElementById('feedbackButtons');
    const feedbackThanks = document.getElementById('feedbackThanks');

    // Disable buttons immediately
    const buttons = feedbackButtons.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => {
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.style.cursor = 'not-allowed';
    });

    // Submit feedback
    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            feedback: feedbackType,
            session_id: sessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            // Hide buttons and show thank you message
            feedbackButtons.style.display = 'none';
            feedbackThanks.style.display = 'flex';
        } else {
            // Re-enable buttons on error
            buttons.forEach(btn => {
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
            });
            alert('Error submitting feedback. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        });
        alert('Error submitting feedback. Please try again.');
    });
}

// Tips toggle
function toggleTips() {
    const button = document.querySelector('.tips-toggle');
    const content = document.getElementById('tipsContent');

    button.classList.toggle('expanded');
    content.classList.toggle('expanded');
}

// Particle animation
function createParticles() {
    const particleCount = 40;
    const hero = document.querySelector('.hero');
    const colors = ['blue', 'purple', 'cyan', 'pink'];

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        const colorClass = colors[Math.floor(Math.random() * colors.length)];
        const shouldGlow = Math.random() > 0.5;

        particle.className = `particle ${colorClass}${shouldGlow ? ' glow' : ''}`;
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 8 + 's';
        particle.style.animationDuration = (Math.random() * 4 + 4) + 's';

        hero.appendChild(particle);
    }

    // Add some larger glowing orbs
    for (let i = 0; i < 5; i++) {
        const orb = document.createElement('div');
        orb.className = 'particle glow';
        orb.style.width = '6px';
        orb.style.height = '6px';
        orb.style.left = Math.random() * 100 + '%';
        orb.style.top = Math.random() * 100 + '%';
        orb.style.animationDelay = Math.random() * 8 + 's';
        orb.style.animationDuration = (Math.random() * 6 + 6) + 's';

        const orbColors = ['#60a5fa', '#a855f7', '#22d3ee', '#ec4899'];
        const orbColor = orbColors[Math.floor(Math.random() * orbColors.length)];
        orb.style.background = orbColor;
        orb.style.color = orbColor;
        orb.style.boxShadow = `0 0 20px ${orbColor}, 0 0 40px ${orbColor}`;

        hero.appendChild(orb);
    }
}

// DOM ready and event listeners
document.addEventListener('DOMContentLoaded', function() {
    // File input and drag/drop functionality
    const uploadSection = document.getElementById('uploadSection');
    const fileInput = document.getElementById('file');

    if (uploadSection && fileInput) {
        uploadSection.addEventListener('click', () => fileInput.click());

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });
    }

    // Character counter for text input
    const textContent = document.getElementById('textContent');
    if (textContent) {
        textContent.addEventListener('input', function(e) {
            const charCount = e.target.value.length;
            const charCountElement = document.getElementById('charCount');
            if (charCountElement) {
                charCountElement.textContent = charCount;

                // Update color based on character count
                charCountElement.classList.remove('warning', 'danger');
                if (charCount > 800) {
                    charCountElement.classList.add('danger');
                } else if (charCount > 600) {
                    charCountElement.classList.add('warning');
                }
            }
        });
    }

    // Form submission
    const analysisForm = document.getElementById('analysisForm');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const fileInput = document.getElementById('file');
            const textInput = document.getElementById('textContent');
            const submitBtn = document.getElementById('submitBtn');
            const loadingDiv = document.getElementById('loadingDiv');
            const activeTab = document.querySelector('.tab-button.active').id;

            // Validate inputs
            if (activeTab === 'fileTab') {
                if (!fileInput.files || !fileInput.files[0]) {
                    alert('Please select a file to upload!');
                    return;
                }
            } else {
                const textValue = textInput.value.trim();
                if (!textValue) {
                    alert('Please enter some text to analyze!');
                    return;
                }
                if (textValue.length < 10) {
                    alert('Please enter at least 10 characters for better analysis!');
                    return;
                }
            }

            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Processing...';
            loadingDiv.classList.add('show');

            // Clear any existing results or flash messages
            const existingResult = document.querySelector('.result');
            const flashMessages = document.querySelector('.flash-messages');
            if (existingResult) existingResult.remove();
            if (flashMessages) flashMessages.innerHTML = '';

            // Scroll to loading div
            loadingDiv.scrollIntoView({ behavior: 'smooth' });

            // Create FormData and submit via AJAX
            const formData = new FormData();
            if (activeTab === 'fileTab') {
                formData.append('file', fileInput.files[0]);
            } else {
                formData.append('text_content', textInput.value.trim());
            }

            fetch('/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(html => {
                // Parse the response HTML
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');

                // Hide loading state
                loadingDiv.classList.remove('show');
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Analyze Content';

                // Check for flash messages in response
                const responseFlash = doc.querySelector('.flash-messages');
                if (responseFlash && responseFlash.innerHTML.trim()) {
                    const currentFlash = document.querySelector('.flash-messages');
                    if (currentFlash) {
                        currentFlash.innerHTML = responseFlash.innerHTML;
                    }
                    return; // Don't process result if there's an error
                }

                // Check for result in response
                const responseResult = doc.querySelector('.result');
                if (responseResult) {
                    // Insert the result after the form
                    const form = document.getElementById('analysisForm');
                    form.insertAdjacentElement('afterend', responseResult);

                    // Scroll to result
                    responseResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error:', error);

                // Hide loading state
                loadingDiv.classList.remove('show');
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Analyze Content';

                // Show error message
                const flashDiv = document.querySelector('.flash-messages');
                if (flashDiv) {
                    flashDiv.innerHTML = `
                        <div class="flash-message flash-error">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>
                            </svg>
                            An error occurred while analyzing your file. Please try again.
                        </div>
                    `;
                }
            });
        });
    }

    // Add entrance animations
    const heroContent = document.querySelector('.hero-content');
    if (heroContent) {
        heroContent.style.opacity = '0';
        heroContent.style.transform = 'translateY(30px)';

        setTimeout(() => {
            heroContent.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
            heroContent.style.opacity = '1';
            heroContent.style.transform = 'translateY(0)';
        }, 200);
    }
});

// Initialize particles when page loads
window.addEventListener('load', function() {
    createParticles();
});

// Expose functions to global scope for inline event handlers
window.scrollToApp = scrollToApp;
window.toggleMobileMenu = toggleMobileMenu;
window.switchTab = switchTab;
window.submitFeedback = submitFeedback;
window.toggleTips = toggleTips;
