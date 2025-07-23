const chatDiv = document.getElementById('chat')!;
const questionInput = document.getElementById('question') as HTMLInputElement;
const askBtn = document.getElementById('ask-btn')!;

function appendMessage(text: string, sender: 'user' | 'bot') {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  bubble.textContent = text;
  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

askBtn.addEventListener('click', askQuestion);

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') askQuestion();
});

function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, 'user');
  questionInput.value = '';
  appendMessage('Thinking...', 'bot');

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0].id) {
      chrome.tabs.sendMessage(
        tabs[0].id,
        { type: 'GET_PAGE_TEXT' },
        async (response) => {
          if (response?.text) {
            try {
              const res = await fetch('http://localhost:5000/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: response.text, question }),
              });
              const data = await res.json();
              // Remove the last "Thinking..." bubble
              const bubbles = chatDiv.getElementsByClassName('bot');
              if (bubbles.length > 0) chatDiv.removeChild(bubbles[bubbles.length - 1]);
              appendMessage(data.answer, 'bot');
            } catch (err) {
              const bubbles = chatDiv.getElementsByClassName('bot');
              if (bubbles.length > 0) chatDiv.removeChild(bubbles[bubbles.length - 1]);
              appendMessage('Backend Error: ' + err, 'bot');
            }
          }
        }
      );
    }
  });
}
