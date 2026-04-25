document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://localhost:8000';
    
    // Elements - Prediction
    const predictionForm = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const resultPrice = document.getElementById('result-price');
    const resultCategory = document.getElementById('result-category');
    const resultInterpretation = document.getElementById('result-interpretation');

    // Elements - Chat
    const chatToggle = document.getElementById('chat-toggle');
    const chatWidget = document.getElementById('chat-widget');
    const closeChat = document.getElementById('close-chat');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendChat = document.getElementById('send-chat');

    // --- Metadata Population ---
    const populateMetadata = async () => {
        try {
            const response = await fetch(`${API_BASE}/metadata`);
            const data = await response.json();
            
            const citySelect = document.getElementById('city');
            const zipSelect = document.getElementById('statezip');
            
            citySelect.innerHTML = data.cities.map(c => `<option value="${c}">${c}</option>`).join('');
            zipSelect.innerHTML = data.zips.map(z => `<option value="${z}">${z}</option>`).join('');
        } catch (error) {
            console.error('Error fetching metadata:', error);
        }
    };

    populateMetadata();

    // --- Prediction Logic ---
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span>Analyzing...</span> <i class="fas fa-spinner fa-spin"></i>';
        
        const formData = new FormData(predictionForm);
        const data = {};
        formData.forEach((value, key) => {
            // Keep city and statezip as strings, others as numbers
            if (key === 'city' || key === 'statezip') {
                data[key] = value;
            } else {
                data[key] = parseFloat(value);
            }
        });

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error('Prediction failed');

            const result = await response.json();
            
            // Update UI
            resultPrice.textContent = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(result.predicted_price);
            
            resultCategory.textContent = result.category;
            resultInterpretation.innerHTML = marked.parse(result.interpretation);
            
            resultContainer.classList.remove('hidden');
            resultContainer.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error(error);
            alert('Error communicating with the backend. Make sure app.py is running.');
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<span>Generate Valuation</span> <i class="fas fa-magic"></i>';
        }
    });

    // --- Chat Logic ---
    chatToggle.addEventListener('click', () => {
        chatWidget.classList.toggle('collapsed');
    });

    closeChat.addEventListener('click', () => {
        chatWidget.classList.add('collapsed');
    });

    const addMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        msgDiv.innerHTML = marked.parse(text);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const handleSendMessage = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        chatInput.value = '';

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) throw new Error('Chat failed');

            const result = await response.json();
            addMessage(result.response, 'bot');
        } catch (error) {
            addMessage("I'm sorry, I'm having trouble connecting to the server.", 'bot');
        }
    };

    sendChat.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSendMessage();
    });
});
