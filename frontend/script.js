async function sendMessage() {
  const input = document.getElementById("userInput");
  const chatbox = document.getElementById("chatbox");
  const question = input.value;

  chatbox.innerHTML += `<p><b>You:</b> ${question}</p>`;
  input.value = "";

  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  const data = await response.json();
  chatbox.innerHTML += `<p><b>Bot:</b> ${data.answer}</p>`;
  chatbox.scrollTop = chatbox.scrollHeight;
}
