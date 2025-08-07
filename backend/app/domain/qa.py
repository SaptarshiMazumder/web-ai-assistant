from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Set
import re
from langgraph.graph import StateGraph, END
from app.utils.text import clean_markdown
from app.config import config
import os
from google.generativeai import configure, GenerativeModel
from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Set up API keys and environment
configure(api_key=config.GOOGLE_API_KEY)
openai_api_key = config.OPENAI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_APPLICATION_CREDENTIALS

DEFAULT_MAX_HOPS = 3
DEFAULT_K_LINKS = 5
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TOTAL_PAGE_BUDGET = 25

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
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
    result = llm.invoke([{"role": "user", "content": prompt}])
    answer_full = (result.content or "").strip()
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
    state.used_chunks = []
    state.sufficient = sufficient
    state.confidence = confidence
    return state

def gemini_answer_node(state: State) -> State:
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
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        answer_full += chunk.text or ""
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
    except Exception:
        pass
    state.answer = answer
    state.used_chunks = []
    state.sufficient = sufficient
    state.confidence = confidence
    return state

async def smart_qa_runner(
    init_state,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    k_links: int = DEFAULT_K_LINKS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    total_page_budget: int = DEFAULT_TOTAL_PAGE_BUDGET,
):
    question = init_state.question
    original_domain = init_state.original_domain
    if not original_domain:
        from urllib.parse import urlparse
        original_domain = urlparse(init_state.page_url).netloc if init_state.page_url else ""
    visited: Set[str] = set(init_state.visited_urls or [])
    hops_used = init_state.hops or 0
    queue = [{
        "page_url": init_state.page_url,
        "text": init_state.text,
        "links": init_state.links or [],
        "depth": 0
    }]
    best_partial = None
    pages_seen = 0
    while queue:
        node = queue.pop(0)
        page_url = node["page_url"]
        text = node["text"]
        links = node["links"]
        depth = node["depth"]
        if page_url:
            visited.add(page_url)
        if pages_seen >= total_page_budget:
            break
        pages_seen += 1
        # 1) Run QA on this page (infrastructure call, to be injected)
        qa_result = None  # Placeholder
        sufficient = qa_result.sufficient if qa_result else False
        # Keep best partial (in case we never get a sufficient one)
        if qa_result and (best_partial is None or (qa_result.confidence or 0) > (getattr(best_partial, 'confidence', 0) or 0)):
            best_partial = qa_result
        if sufficient and qa_result:
            return {
                "answer": getattr(qa_result, 'answer', ""),
                "sources": getattr(qa_result, 'sources', []),
                "visited_urls": list(visited),
                "sufficient": True
            }
        if depth >= max_hops:
            continue
        # Filter links (same domain, unvisited)
        from urllib.parse import urlparse
        candidate_links = [
            l for l in links
            if l.get("href", "").startswith("http")
            and l["href"] not in visited
            and urlparse(l["href"]).netloc == original_domain
        ]
        if not candidate_links:
            continue
        # 2) LLM link selection and scraping are infrastructure, to be injected
        selected_links = []  # Placeholder
        child_results = []   # Placeholder
        # 3) Check if any are sufficient → early stop
        for cr in child_results:
            if getattr(cr, 'url', None):
                visited.add(cr.url)
            if getattr(cr, 'sufficient', False):
                return {
                    "answer": getattr(cr, 'answer', ""),
                    "sources": getattr(cr, 'sources', []),
                    "visited_urls": list(visited),
                    "sufficient": True
                }
        # 4) Otherwise, push children (insufficient ones) back into the queue (BFS)
        for cr in child_results:
            if getattr(cr, 'url', None):
                visited.add(cr.url)
            queue.append({
                "page_url": getattr(cr, 'url', None),
                "text": getattr(cr, 'text', ""),
                "links": getattr(cr, 'links', []),
                "depth": depth + 1
            })
    # If we exit the loop without a sufficient answer:
    if best_partial:
        return {
            "answer": getattr(best_partial, 'answer', ""),
            "sources": getattr(best_partial, 'sources', []),
            "visited_urls": list(visited),
            "sufficient": False,
            "confidence": getattr(best_partial, 'confidence', None),
            "multi_page": False
        }
    else:
        return {
            "answer": "Sorry, I couldn't find a sufficient answer on the pages I visited.",
            "sources": [],
            "visited_urls": list(visited),
            "sufficient": False
        }

# LangGraph setup — just one node now!
qa_builder = StateGraph(State)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("Answer")
smart_qa_graph = qa_builder.compile()