document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('video-file');
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    document.getElementById('upload-status').innerText = result.success ? 'File uploaded and processed successfully!' : `Error: ${result.error}`;
});

document.getElementById('send-button').addEventListener('click', async function () {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value;
    
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    const result = await response.json();
    const chatBox = document.getElementById('chat-box');
    
    chatBox.innerHTML += `<p><strong>You:</strong> ${message}</p>`;
    chatBox.innerHTML += `<p><strong>Bot:</strong> ${result.response}</p>`;
    
    chatInput.value = '';
});
