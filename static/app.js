document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const statusArea = document.querySelector('.status-area');
    const progressFill = document.querySelector('.progress-fill');
    const statusText = document.getElementById('status-text');
    const resultsArea = document.querySelector('.results-area');
    const resultLabel = document.getElementById('result-label');
    const confidenceValue = document.getElementById('confidence-value');
    const resultBadge = document.getElementById('result-badge');
    const probabilityBar = document.getElementById('probability-bar');
    const spectrogramDisplay = document.getElementById('spectrogram-display');
    const recordBtn = document.getElementById('record-btn');
    const recordText = document.getElementById('record-text');
    const researchLink = document.getElementById('research-link');
    const researchSection = document.getElementById('research-section');
    const spectrogramContainer = document.querySelector('.spectrogram-container');

    let audioContext;
    let processor;
    let source;
    let audioData = [];
    let isRecording = false;

    // Click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Toggle Research
    researchLink.addEventListener('click', () => {
        const isHidden = researchSection.style.display === 'none' || researchSection.style.display === '';
        researchSection.style.display = isHidden ? 'block' : 'none';
        researchLink.style.color = isHidden ? 'var(--text)' : 'var(--text-dim)';
        if (isHidden) researchSection.scrollIntoView({ behavior: 'smooth' });
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragging'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragging'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }

    // --- Recording Logic (PCM to WAV) ---
    recordBtn.addEventListener('click', async () => {
        if (isRecording) {
            stopRecording();
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
            source = audioContext.createMediaStreamSource(stream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            audioData = [];
            
            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                audioData.push(new Float32Array(inputData));
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            isRecording = true;
            recordBtn.classList.add('active');
            recordText.innerText = 'Stop Recording (3s)...';

            // Auto-stop after 3 seconds
            setTimeout(() => {
                if (isRecording) stopRecording();
            }, 3000);

        } catch (err) {
            console.error("Microphone access denied:", err);
            alert("Could not access microphone. Please check permissions.");
        }
    });

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        
        recordBtn.classList.remove('active');
        recordText.innerText = 'Record Voice';

        // Disconnect nodes
        processor.disconnect();
        source.disconnect();
        if (audioContext.state !== 'closed') audioContext.close();

        // Flatten audioData
        const flattened = flattenArray(audioData);
        const wavBlob = encodeWAV(flattened, 22050);
        const file = new File([wavBlob], "recorded_audio.wav", { type: "audio/wav" });
        
        uploadFile(file);
    }

    function flattenArray(channelBuffer) {
        let result = new Float32Array(channelBuffer.reduce((acc, b) => acc + b.length, 0));
        let offset = 0;
        for (let i = 0; i < channelBuffer.length; i++) {
            result.set(channelBuffer[i], offset);
            offset += channelBuffer[i].length;
        }
        return result;
    }

    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 32 + samples.length * 2, true);
        /* RIFF type */
        writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw) */
        view.setUint16(20, 1, true);
        /* channel count */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sample rate * block align) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (channel count * bytes per sample) */
        view.setUint16(32, 2, true);
        /* bits per sample */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Write PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            let s = Math.max(-1, min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function min(a, b) { return a < b ? a : b; }

    async function uploadFile(file) {
        // Reset UI
        resultsArea.style.display = 'none';
        statusArea.style.display = 'block';
        statusText.innerText = `Analyzing ${file.name}...`;
        progressFill.style.width = '30%';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            progressFill.style.width = '70%';

            if (response.ok) {
                const data = await response.json();
                showResults(data);
            } else {
                const errorData = await response.json();
                alert(`Error: ${errorData.error || 'Server error'}`);
                statusArea.style.display = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Could not connect to the server.');
            statusArea.style.display = 'none';
        }
    }

    function showResults(data) {
        statusArea.style.display = 'none';
        resultsArea.style.display = 'block';

        resultLabel.innerText = data.label;
        confidenceValue.innerText = `${(data.confidence * 100).toFixed(1)}%`;
        
        // Update badge
        resultBadge.className = 'result-badge';
        if (data.label === "Parkinson's Disease") {
            resultBadge.classList.add('badge-parkinsons');
            resultBadge.innerText = 'High Risk Detected';
        } else {
            resultBadge.classList.add('badge-healthy');
            resultBadge.innerText = 'Neuro-Healthy';
        }

        // Update probability bar
        probabilityBar.style.width = `${data.probability * 100}%`;

        // Update Spectrogram
        if (data.spectrogram) {
            spectrogramContainer.style.display = 'block';
            spectrogramDisplay.src = `data:image/png;base64,${data.spectrogram}`;
        } else {
            spectrogramContainer.style.display = 'none';
            spectrogramDisplay.src = '';
        }
    }
});
