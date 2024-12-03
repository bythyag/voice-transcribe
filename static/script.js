document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('recordButton');
    const transcriptionArea = document.getElementById('transcription');
    let isRecording = false;
    let eventSource = null;

    recordButton.addEventListener('click', () => {
        if (!isRecording) {
            // Start recording
            fetch('/start_recording', { method: 'POST' })
                .then(() => {
                    isRecording = true;
                    recordButton.textContent = 'Stop Recording';
                    
                    // Setup SSE for transcription
                    eventSource = new EventSource('/transcription_stream');
                    eventSource.onmessage = (event) => {
                        if (event.data.trim()) {
                            transcriptionArea.value += event.data + '\n';
                            transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
                        }
                    };
                });
        } else {
            // Stop recording
            fetch('/stop_recording', { method: 'POST' })
                .then(() => {
                    isRecording = false;
                    recordButton.textContent = 'Record';
                    
                    // Close event source
                    if (eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                });
        }
    });
});