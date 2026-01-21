import os, json, re, asyncio
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from utils import log_llm_prompt
from logging_relay import smartqa_log_relay
from google.generativeai import configure, GenerativeModel
from google import genai
from google.genai import types
from config import config
from state import AssistantState
import uuid

configure(api_key=config.GOOGLE_API_KEY)
openai_api_key = config.OPENAI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_APPLICATION_CREDENTIALS

# Backend flag to control whether partial answers are pushed over websocket.
# Set to False to avoid duplicate answers in the UI when the final HTTP
# response also renders the answer. When True, streaming deltas are sent.
ENABLE_ANSWER_STREAMING = True

def clean_markdown(md: str) -> str:
    md = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md)
    md = re.sub(r'http[s]?://\S+', '', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()

async def webpage_answer_node(state: BaseModel) -> BaseModel:
    page_text = state.text
    question = state.question
    page_url = getattr(state, "page_url", "")
    clean_text = clean_markdown(page_text)
    # Stage: planning/analysis instruction prompt build
    try:
        smartqa_log_relay.log(json.dumps({
            "type": "stage",
            "stage": "node_start",
            "node": "webpage_answer",
            "page_url": page_url,
        }))
    except Exception:
        pass

    prompt = (
        "You are an expert assistant. Use only the current page's content below. Answer by quoting the relevant passage, code block, or table WORD-FOR-WORD, including ALL formatting, indentation, and line breaks. "
        "Do not paraphrase, summarize, or shorten quoted material unless absolutely necessary. "
        "Prioritize giving the most detailed answer possible by quoting all relevant text from the page. "
        "If the page has numbers, prices, code, tables, or other specifics, quote them exactly. "
        "If multiple relevant passages are found, include ALL of them word-for-word. "
        "If the question asks for a figure, comparison, change, count, average, percentage, or similar, also provide a DIRECT ANSWER (bullets or a small table are OK).\n"
        "If you cite a source, include the full URL as shown in the source. "
        "If the answer is not on this page, briefly say that the information isn't on this page {page_url}. Do not apologize. Do not use phrases like 'provided content' or 'the provided context'. "
        "Do not hallucinate. "
        "At the end, write 'SUFFICIENT: YES' if the answer fully resolves the question, or 'SUFFICIENT: NO' if not. "
        "Write 'CONFIDENCE: <0-100>%'. "
        f"CONTENT:\n{clean_text}\n\n"
        f"USER QUESTION: {question}\n"
        f"(Page URL: {page_url})"
    )
    log_llm_prompt(prompt)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=1)
    # Stream tokens to frontend via SmartQA log websocket
    full_text_accum = ""
    streamed_answer_sent_len = 0
    stream_id = page_url or f"stream-{uuid.uuid4().hex}"
    try:
        if ENABLE_ANSWER_STREAMING:
            smartqa_log_relay.log(json.dumps({"type": "answer_reset", "stream_id": stream_id}))
            smartqa_log_relay.log(json.dumps({
                "type": "stage",
                "stage": "generation_stream_start",
                "node": "webpage_answer",
                "page_url": page_url,
                "stream_id": stream_id,
            }))
        for chunk in llm.stream([{ "role": "user", "content": prompt }]):
            try:
                delta_text = getattr(chunk, "content", None)
                if not delta_text:
                    # Yield control to allow websocket sender to flush any pending messages
                    await asyncio.sleep(0)
                    continue
                full_text_accum += delta_text
                # Exclude footer/meta markers (e.g., SUFFICIENT/CONFIDENCE/FULL INFO) from streamed content
                m_any = re.search(r"\b(SUFFICIENT|CONFIDENCE|FULL\s*INFO)\s*:", full_text_accum, flags=re.IGNORECASE)
                cut_idxs = [m_any.start()] if m_any else []
                cut_idx = min(cut_idxs) if cut_idxs else len(full_text_accum)
                display_text = full_text_accum[:cut_idx]
                if ENABLE_ANSWER_STREAMING and len(display_text) > streamed_answer_sent_len:
                    delta_to_send = display_text[streamed_answer_sent_len:]
                    smartqa_log_relay.log(json.dumps({
                        "type": "answer_delta",
                        "text": delta_to_send,
                        "stream_id": stream_id,      
                    }))
                    streamed_answer_sent_len = len(display_text)
                # Yield to the event loop so websocket messages can be sent immediately
                await asyncio.sleep(0)
            except Exception:
                # Still yield to avoid starving the loop
                await asyncio.sleep(0)
                pass
    except Exception:
        # Fallback to single-shot if streaming not available
        result = llm.invoke([{ "role": "user", "content": prompt }])
        full_text_accum = (result.content or "").strip()

    answer_full = full_text_accum.strip()
    # Extract SUFFICIENT marker anywhere in the text (not just after a newline)
    m_sufficient = re.search(r'\bSUFFICIENT\s*:\s*(YES|NO)\b', answer_full, flags=re.IGNORECASE)
    if m_sufficient:
        answer = answer_full[:m_sufficient.start()].rstrip()
        sufficient = (m_sufficient.group(1).upper() == 'YES')
    else:
        answer = answer_full
        sufficient = False
    # Extract confidence anywhere in the text
    confidence_match = re.search(r'\bCONFIDENCE\s*:\s*(\d+)%', answer_full, flags=re.IGNORECASE)
    confidence = int(confidence_match.group(1)) if confidence_match else 0
    try:
        if ENABLE_ANSWER_STREAMING:
            smartqa_log_relay.log(json.dumps({
                "type": "answer_done",
                "sufficient": sufficient,
                "confidence": confidence,
                "stream_id": stream_id,  
            }))
        smartqa_log_relay.log(json.dumps({
            "type": "stage",
            "stage": "node_done",
            "node": "webpage_answer",
            "page_url": page_url,
            "sufficient": sufficient,
            "confidence": confidence,
        }))
    except Exception:
        pass
    # answer = answer_full
    state.answer = answer
    state.used_chunks = []
    state.sufficient = sufficient
    state.confidence = confidence

    
    return state

def google_search_answer_node(state):
    print("🔎 Using Gemini grounded answer node")
    PROJECT_ID = config.PROJECT_ID
    LOCATION = config.LOCATION
    question = state.question
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )
    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    gen_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
        ],
        tools=tools,
    )
    answer_full = ""
    response = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=gen_config,
    ):
        response = chunk
        print("Chunk:", chunk)
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        answer_full += chunk.text or ""
    print("\n\n====================== ANSWER ======================")
    print(answer_full)
    print("====================================================")
    confidence_match = re.search(r'CONFIDENCE: (\d+)%', answer_full)
    confidence = int(confidence_match.group(1)) if confidence_match else None
    if "\nSUFFICIENT: YES" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: YES")[0].strip()
        sufficient = True
    elif "\nSUFFICIENT: NO" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: NO")[0].strip()
        sufficient = False
    else:
        answer = answer_full
        sufficient = False
    try:
        grounding_metadata = response.candidates[0].grounding_metadata
        if grounding_metadata and grounding_metadata.grounding_chunks:
            answer += "\n\n🔗 Sources:\n"
            for i, chunk in enumerate(grounding_metadata.grounding_chunks, 1):
                context = chunk.web or chunk.retrieved_context
                if not context or not context.uri:
                    continue
                uri = context.uri
                title = context.title or "Source"
                snippet = getattr(context, "snippet", None)
                if uri.startswith("gs://"):
                    uri = uri.replace("gs://", "https://storage.googleapis.com/", 1).replace(" ", "%20")
                answer += f"{i}. [{title}]({uri})\n"
                if snippet:
                    answer += f"    → {snippet.strip()}\n"
    except Exception as e:
        print("⚠️ Failed to extract grounding sources:", str(e))
    state.answer = answer
    state.used_chunks = []
    state.sufficient = sufficient
    state.confidence = confidence
    return state


