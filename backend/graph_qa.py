import os, json, re
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from urllib.parse import urlparse
from logging_relay import log, smartqa_log_relay
from utils import log_llm_prompt
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()


configure(api_key=os.environ.get("GOOGLE_API_KEY"))
openai_api_key = os.environ.get("OPENAI_API_KEY")

class State(BaseModel):
    text: str
    question: str
    enhanced_query: str = ""
    docs: List[Any] = []
    retrieved_docs: List[Any] = []
    answer: str = ""
    used_chunks: List[Dict[str, Any]] = []
    page_url: str = ""
    sufficient: bool = False
    confidence: Optional[int] = None

# def enhance_query_node(state: State) -> State:
#     page_text = state.text
#     user_question = state.question
#     prompt = (
#         "You are a query rewriter for search. Only improve or clarify the question if needed, and ONLY use phrases or words that appear in the provided web page content. "
#         "If the question is already good, return it unchanged. NEVER invent new topics, courses, or details not present in the page.\n\n"
#         f"WEBPAGE CONTENT (first 3000 chars):\n{page_text[:3000]}\n\n"
#         f"USER QUESTION: {user_question}\n\n"
#         "REWRITTEN QUERY:"
#     )
#     llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
#     result = llm.invoke([{"role": "user", "content": prompt}])
#     enhanced_query = result.content.strip()
#     state.enhanced_query = enhanced_query
#     return state




def answer_node(state: State) -> State:
    page_text = state.text
    question = state.question
    page_url = getattr(state, "page_url", "")

    clean_text = clean_markdown(page_text)
    prompt = (
       "You are an expert assistant. Using only the current webpage content below, answer the user's question by quoting the relevant passage, code block, or table WORD-FOR-WORD, including ALL formatting, indentation, and line breaks. "
        "DO NOT paraphrase, summarize, or shorten ANY part of the quoted answer, unless absolutely necessary. "
        "Prioritize giving the most detailed answer possible by quoting the all relevant text from the content. "
        "If the content has numbers, prices, code, tables, or other specific details, quote them exactly as they appear. "
        "If multiple relevant passages are found, include ALL of them in their entirety, word-for-word. "
        "ALWAYS provide which page they should visit to find the exact information they are looking for, if possible, by saying 'Please visit [page URL] for the full details.' "
        "If you cite information from a source, always include the full URL shown in the source."
        "If no answer is found, summarize anything related, and politely inform the user that the answer does not appear to be present, suggest a related page by URL if present in the content. "
        "Do not hallucinate."
        "At the end, write 'SUFFICIENT: YES' if the answer fully resolves the question, or 'SUFFICIENT: NO' if not."
        " Write 'CONFIDENCE: <0-100>%, based on how confident you are that this answers the question fully'."
        " Write 'Full info or more info: <URL>' \n\n"
        f"CONTENT:\n{clean_text}\n\n"
        f"USER QUESTION: {question}\n"
        f"(Page URL: {page_url})"
    )
    # === LOGGING PROMPT ===
    log_llm_prompt(prompt)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
    result = llm.invoke([{"role": "user", "content": prompt}])
    answer_full = (result.content or "").strip()

    # model = GenerativeModel("models/gemini-1.5-flash")
    # response = model.generate_content(prompt)
    # answer_full = (response.text or "").strip()

    confidence_match = re.search(r'CONFIDENCE: (\d+)%', answer_full)
    confidence = int(confidence_match.group(1)) if confidence_match else 0

    if "\nSUFFICIENT: YES" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: YES")[0].strip()
        sufficient = True
    elif "\nSUFFICIENT: NO" in answer_full:
        answer = answer_full.split("\nSUFFICIENT: NO")[0].strip()
        sufficient = False
    else:
        answer = answer_full
        sufficient = False

    state.answer = answer
    state.used_chunks = []  # No more chunks!
    state.sufficient = sufficient
    state.confidence = confidence
    return state


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\googler\Downloads\gen-lang-client-0545494042-a92b6b867500.json"  # 🔁 your JSON key here
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\googler\Downloads\tour-proj-451201-f03b91fdf3d7.json"

def gemini_answer_node(state):
    print("🔎 Using Gemini grounded answer node")
    PROJECT_ID = "gen-lang-client-0545494042"
    PROJECT_ID = "tour-proj-451201"

    question = state.question

    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,  # 🔁 Replace with your actual GCP project ID
        location="us-central1"          # or "europe-west4", "global"
    )

    model = "gemini-2.0-flash-001"  # 🔁 use "gemini-1.5-pro-001" or "gemini-2.5-pro" if enabled

    # Create prompt
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    ]

    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
        ],
        tools=tools,
        # thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    answer_full = ""
    # Store the full response for post-processing
    response = None
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        response = chunk  # Save the final full chunk to get grounding info
        print("Chunk:", chunk)
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        answer_full += chunk.text or ""
    print("\n\n====================== ANSWER ======================")
    print(answer_full)
    print("====================================================")
    # === Parse answer for confidence and sufficiency ===
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

                # Convert gs:// to real link
                if uri.startswith("gs://"):
                    uri = uri.replace("gs://", "https://storage.googleapis.com/", 1).replace(" ", "%20")

                answer += f"{i}. [{title}]({uri})\n"
                if snippet:
                    answer += f"    → {snippet.strip()}\n"
    except Exception as e:
        print("⚠️ Failed to extract grounding sources:", str(e))





    # === Update state ===
    state.answer = answer
    state.used_chunks = []
    state.sufficient = sufficient
    state.confidence = confidence

    return state


# LangGraph setup — just one node now!
qa_builder = StateGraph(State)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("Answer")
qa_graph = qa_builder.compile()

# new logic
def clean_markdown(md: str) -> str:
    md = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md)
    md = re.sub(r'http[s]?://\S+', '', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()



qa_builder = StateGraph(State)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("Answer")
qa_graph = qa_builder.compile()
