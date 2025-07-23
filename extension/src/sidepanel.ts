import { marked } from "marked";
import hljs from "highlight.js";

// TypeScript global augmentation (only needed if you see errors)
declare global {
  interface Window {
    pageText?: string; // or however you store page text
  }
}

const chatDiv = document.getElementById("chat")!;
const askBtn = document.getElementById("ask-btn")!;
const questionInput = document.getElementById("question")! as HTMLInputElement;

async function appendMessage(text: string, sender: "user" | "bot") {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + sender;
  bubble.innerHTML = await marked.parse(text);

  // Highlight all code blocks (after innerHTML set)
  bubble.querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block as HTMLElement);
  });

  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

askBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, "user");

  // Example: get page text however your extension does it (here using window.pageText)
  const pageText = window.pageText || "";

  // Send question to backend
  const resp = await fetch("http://localhost:5000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: pageText, question }),
  });
  const data = await resp.json();

  // data.answer is markdown
  appendMessage(data.answer, "bot");
  questionInput.value = "";
};
