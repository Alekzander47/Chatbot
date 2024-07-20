document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.querySelector('.chat-container');
    const darkModeToggle = document.querySelector('#dark-mode-toggle');
    const chatInput = document.querySelector('#chat-input');
    const sendButton = document.querySelector('#send-button');
    const attachButton = document.querySelector('#attach-button');
    const fileInput = document.querySelector('#file-input');
    const sidebar = document.querySelector('#sidebar');
    const sidebarToggle = document.querySelector('#sidebar-toggle');
    const sidebarClose = document.querySelector('#sidebar-close');
    const loadingIndicator = document.querySelector('#loading-indicator');
    const documentList = document.querySelector('#document-list');
    
    const userImageUrl = window.userImageUrl; // Get the URL defined in HTML
    const assistantImageUrl = window.assistantImageUrl; // Get the URL defined in HTML
    
    // Dark mode toggle
    darkModeToggle.addEventListener('click', () => {
        chatContainer.classList.toggle('dark-mode');
    });
    
    // File input for attachments
    attachButton.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (event) => {
        const files = event.target.files;
        displayUploadedFiles(files);
        uploadFiles(files);
    });
    
    // Send message
    sendButton.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message !== '') {
            loadingIndicator.classList.remove('hidden'); // Show loading indicator

            fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ message })
            })
            .then(response => response.json())
            .then(data => {
                // Create user bubble with profile image
                const userBubbleContainer = document.createElement('div');
                userBubbleContainer.classList.add('bubble-container', 'user-bubble-container');

                const userProfileImage = document.createElement('img');
                userProfileImage.src = userImageUrl; // Use the variable
                userProfileImage.classList.add('profile-image');
                
                const userBubble = document.createElement('div');
                userBubble.classList.add('message-bubble', 'user-bubble');
                userBubble.textContent = message;

                userBubbleContainer.appendChild(userProfileImage);
                userBubbleContainer.appendChild(userBubble);

                // Create assistant bubble with profile image
                const assistantBubbleContainer = document.createElement('div');
                assistantBubbleContainer.classList.add('bubble-container', 'assistant-bubble-container');

                const assistantProfileImage = document.createElement('img');
                assistantProfileImage.src = assistantImageUrl; // Use the variable
                assistantProfileImage.classList.add('profile-image');
                
                const assistantBubble = document.createElement('div');
                assistantBubble.classList.add('message-bubble', 'assistant-bubble');
                assistantBubble.textContent = data.response;

                assistantBubbleContainer.appendChild(assistantProfileImage);
                assistantBubbleContainer.appendChild(assistantBubble);

                // Append user bubble container first, then assistant bubble container
                const chatMessages = document.querySelector('.chat-messages');
                chatMessages.appendChild(userBubbleContainer);
                chatMessages.appendChild(assistantBubbleContainer);
                chatMessages.appendChild(document.createElement('div')).classList.add('clearfix');

                chatInput.value = '';
                loadingIndicator.classList.add('hidden'); // Hide loading indicator
            })
            .catch(error => {
                console.error('Error:', error);
                loadingIndicator.classList.add('hidden'); // Hide loading indicator on error
            });
        }
    });

    // Sidebar toggle functionality
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
    });

    sidebarClose.addEventListener('click', () => {
        sidebar.classList.add('hidden');
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    function displayUploadedFiles(files) {
        documentList.innerHTML = ''; // Clear previous list
        Array.from(files).forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.classList.add('file-item');
            fileItem.dataset.fileName = file.name;

            const fileName = document.createElement('span');
            fileName.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;

            const removeButton = document.createElement('button');
            removeButton.textContent = 'X';
            removeButton.addEventListener('click', () => {
                fileItem.remove();
            });

            fileItem.appendChild(fileName);
            fileItem.appendChild(removeButton);
            documentList.appendChild(fileItem);
        });
    }

    function uploadFiles(files) {
        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('uploaded_files', file);
        });

        fetch('/upload-files/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Files uploaded successfully:', data);
        })
        .catch(error => {
            console.error('Error uploading files:', error);
        });
    }
});
