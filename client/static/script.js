document.addEventListener('DOMContentLoaded', () => {
    // -------------------- DOM ELEMENTS (Scoped where possible) --------------------
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatMicButton = document.getElementById('mic-button');
    const inputArea = document.getElementById('input-area');
    const startStopButton = document.getElementById('start-stop-button');
    const videoFeed = document.getElementById('video-feed');
    const cameraPreview = document.getElementById('camera-preview');
    const previewVideo = document.getElementById('preview-video');
    const audioToggleCheckbox = document.getElementById('audio-toggle-checkbox');
    const cameraPreviewToggle = document.getElementById('camera-preview-toggle');
    const audioVisualizer = document.getElementById('audio-visualizer');
    const monitorStateWrapper = document.getElementById('monitor-state-wrapper');

    // -------------------- STATE --------------------
    let mediaStream = null;
    let previewStream = null;
    let monitorMediaRecorder;
    let videoChunks = [];
    let isMonitorRecording = false;
    let audioContext, analyser, microphoneSource, audioDataArray, animationFrameId;
    let isChatRecording = false;

    const BACKEND_URL = "http://127.0.0.1:5000/api"; // CHANGE: Pointing to the /api prefix

    // -------------------- UTILITY --------------------
const addMessage = (content, sender) => {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', `${sender}-message`);

    if (sender === 'bot') {
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        messageContainer.appendChild(avatar);
    }

    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');

    if (typeof content === 'string' || typeof content === 'number') {
        // Handle plain text
        const p = document.createElement('p');
        p.textContent = content;
        messageContent.appendChild(p);
    } else if (content instanceof Node) {
        // Handle DOM nodes like <video>, <img>, etc.
        messageContent.appendChild(content);
    } else if (content) {
        // Handle objects/arrays safely
        const p = document.createElement('p');
        p.textContent = JSON.stringify(content);
        messageContent.appendChild(p);
    } else {
        // Handle null/undefined gracefully
        const p = document.createElement('p');
        p.textContent = "[No response]";
        messageContent.appendChild(p);
    }

    messageContainer.appendChild(messageContent);
    chatLog.appendChild(messageContainer);
    chatLog.scrollTop = chatLog.scrollHeight;
};


    // -------------------- CHATBOT & BACKEND CALLS --------------------
    async function sendTextToChatbot(userText) {
        if (!userText.trim()) return;
        addMessage(userText, 'user');
        userInput.value = '';

        try {
            const response = await fetch(`${BACKEND_URL}/chatbot_text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userText })
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            addMessage(data.response, 'bot');
        } catch (err) {
            console.error('Chatbot error:', err);
            addMessage("Error connecting to the assistant.", 'bot');
        }
    }

    const sendVideoToBackend = async (videoBlob, userText) => {
        const formData = new FormData();
        formData.append('video', videoBlob, 'session.webm');
        formData.append('user_text', userText); // Send user text with the video

        addMessage("Analyzing your session...", 'bot');

        try {
            const response = await fetch(`${BACKEND_URL}/process_multimodal`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const result = await response.json();
            const reply = result.response || result.final_response || JSON.stringify(result);
            addMessage(reply, 'bot');
        } catch (error) {
            console.error('Error sending video to backend:', error);
            addMessage("Sorry, I could not analyze the session.", 'bot');
        }
    };


    // -------------------- SPEECH RECOGNITION (No changes needed) --------------------
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.onstart = () => {
            isChatRecording = true;
            chatMicButton.classList.add('recording');
            inputArea.classList.add('listening');
            userInput.placeholder = "Listening...";
        };
        recognition.onend = () => {
            isChatRecording = false;
            chatMicButton.classList.remove('recording');
            inputArea.classList.remove('listening');
            userInput.placeholder = "Tell me how you feel or press the mic to talk";
        };
        recognition.onerror = (event) => console.error("Speech recognition error:", event.error);
        recognition.onresult = (event) => sendTextToChatbot(event.results[0][0].transcript);
    } else {
        console.log("Web Speech API not available.");
        chatMicButton.style.display = 'none';
    }

    // -------------------- DEVICE MANAGEMENT & DYNAMIC CONTENT --------------------
    function createAudioVisualizerBars() {
        const numberOfBars = 32;
        let barsHTML = '';
        for (let i = 0; i < numberOfBars; i++) {
            barsHTML += '<div class="bar"></div>';
        }
        audioVisualizer.innerHTML = barsHTML;
    }

    async function getDevices() {
        // CHANGE: Device selectors are now scoped inside this function because they are created dynamically.
        const cameraSelect = document.getElementById('camera-select');
        const micSelect = document.getElementById('mic-select');
        if (!cameraSelect || !micSelect) return;

        try {
            await navigator.mediaDevices.getUserMedia({ video: true, audio: true }); // Request permissions
            const devices = await navigator.mediaDevices.enumerateDevices();
            cameraSelect.innerHTML = '';
            micSelect.innerHTML = '';

            devices.filter(d => d.kind === 'videoinput').forEach((device, idx) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${idx + 1}`;
                cameraSelect.appendChild(option);
            });
            devices.filter(d => d.kind === 'audioinput').forEach((device, idx) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Microphone ${idx + 1}`;
                micSelect.appendChild(option);
            });
            cameraSelect.addEventListener('change', updatePreview);

        } catch (error) {
            console.error("Error getting devices:", error);
            monitorStateWrapper.innerHTML = `<p class="error-text">Camera and microphone access is required for the monitor to work. Please grant permissions and refresh the page.</p>`;
            startStopButton.disabled = true;
        }
    }

    function renderInitialMonitorState() {
        const initialStateHTML = `
            <div id="initial-state-container">
                <svg id="placeholder-icon" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><g fill="none" stroke="currentColor" stroke-width="4"><circle cx="100" cy="100" r="70" /><path d="M155 105 A 55 50 0 0 1 45 105" fill="currentColor" opacity="0.2" /><path d="M50 150 Q100 180 150 150 L140 190 Q100 210 60 190 Z" fill="currentColor" opacity="0.4" /><line x1="60" y1="75" x2="40" y2="45" /><circle cx="40" cy="45" r="5" fill="currentColor" /></g></svg>
                <div id="device-selectors">
                    <div class="selector-group"><label for="camera-select">Camera:</label><select id="camera-select"></select></div>
                    <div class="selector-group"><label for="mic-select">Microphone:</label><select id="mic-select"></select></div>
                </div>
            </div>`;
        monitorStateWrapper.innerHTML = initialStateHTML;
        // CHANGE: Populate devices right after creating the dropdowns.
        getDevices();
    }


    // -------------------- VIDEO MONITORING --------------------
    const startMonitorRecording = async () => {
        const cameraSelect = document.getElementById('camera-select'); // Get fresh reference
        const micSelect = document.getElementById('mic-select');     // Get fresh reference

        try {
            const videoConstraints = cameraSelect.value ? { deviceId: { exact: cameraSelect.value } } : true;
            const audioConstraints = audioToggleCheckbox.checked ? (micSelect.value ? { deviceId: { exact: micSelect.value } } : true) : false;

            mediaStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: audioConstraints });

            isMonitorRecording = true;
            videoFeed.srcObject = mediaStream;
            videoFeed.classList.remove('hidden');
            if (audioConstraints) audioVisualizer.classList.remove('hidden');

            // CHANGE: Instead of hiding a static element, we clear the dynamic wrapper.
            monitorStateWrapper.innerHTML = '';

            startStopButton.innerHTML = `<i class="fa-solid fa-stop"></i> Stop`;
            startStopButton.classList.remove('start-button');
            startStopButton.classList.add('stop-button');

            videoChunks = [];
            monitorMediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm' });

            monitorMediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) videoChunks.push(event.data);
            };

            monitorMediaRecorder.onstop = () => {
                const videoBlob = new Blob(videoChunks, { type: 'video/webm' });
                // Pass the current text from the input field to the backend.
                sendVideoToBackend(videoBlob, userInput.value);
                userInput.value = ''; // Clear input after sending
            };

            monitorMediaRecorder.start();
            if (audioConstraints) setupAudioVisualizer();

        } catch (error) {
            console.error("Error starting recording:", error);
        }
    };

    const stopMonitorRecording = () => {
        if (monitorMediaRecorder && monitorMediaRecorder.state === "recording") {
            monitorMediaRecorder.stop();
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
        }
        isMonitorRecording = false;

        videoFeed.classList.add('hidden');
        cameraPreview.classList.add('hidden');
        audioVisualizer.classList.add('hidden');

        // CHANGE: Instead of showing a static element, we re-render the initial state.
        renderInitialMonitorState();

        startStopButton.innerHTML = `<i class="fa-solid fa-play"></i> Start`;
        startStopButton.classList.remove('stop-button');
        startStopButton.classList.add('start-button');

        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };


    // -------------------- CAMERA PREVIEW (No major changes) --------------------
    const togglePreview = async () => { /* Your togglePreview logic here */ };
    const updatePreview = async () => { /* Your updatePreview logic here */ };

    // -------------------- AUDIO VISUALIZER (No major changes) --------------------
    const setupAudioVisualizer = () => { /* Your setupAudioVisualizer logic here */ };
    const visualize = () => { /* Your visualize logic here */ };

    // -------------------- EVENT LISTENERS & INITIALIZATION --------------------
    const handleSendMessage = () => {
        // If monitor is running, stopping it sends video AND text.
        // Otherwise, just send text.
        if (isMonitorRecording) {
            stopMonitorRecording();
        } else {
            sendTextToChatbot(userInput.value);
        }
    };
    
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keydown', (e) => e.key === 'Enter' && handleSendMessage());
    if (recognition) chatMicButton.addEventListener('click', () => isChatRecording ? recognition.stop() : recognition.start());
    startStopButton.addEventListener('click', () => isMonitorRecording ? stopMonitorRecording() : startMonitorRecording());
    cameraPreviewToggle.addEventListener('click', togglePreview);
    audioToggleCheckbox.addEventListener('change', () => { if (mediaStream && mediaStream.getAudioTracks().length > 0) mediaStream.getAudioTracks()[0].enabled = audioToggleCheckbox.checked; });

    // -------------------- INITIALIZE APP --------------------
    function initialize() {
        addMessage("I'm here to help. How are you feeling today?", 'bot');
        createAudioVisualizerBars();
        renderInitialMonitorState();
    }

    initialize();
});
