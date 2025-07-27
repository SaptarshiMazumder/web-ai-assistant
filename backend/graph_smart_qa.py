import os
import json
import re
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from graph_qa import enhance_query_node, retrieve_node, answer_node

openai_api_key = os.environ.get("OPENAI_API_KEY")

class SmartQARequest(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str

class SmartHopState(BaseModel):
    text: str
    question: str
    links: List[Dict[str, str]]
    page_url: str
    answer: str = ""
    sources: List[Any] = []
    sufficient: bool = False
    selected_link: Optional[Dict[str, str]] = None
    visited_urls: List[str] = []
    hops: int = 0
    original_domain: str = ""

def extract_json_from_text(text):
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return arr_match.group(0)
    return None

def answer_sufficiency_llm_node(state):
    prompt = (
        f"Question: {state['question']}\n"
        f"Answer given: {state['answer']}\n\n"
        "Based on the answer, is the user's question fully answered with clear and specific information? "
        "Reply with only 'YES' if it is enough, or 'NO' if it is not clear/specific enough."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    sufficient = "yes" in result.content.strip().lower()
    state["sufficient"] = sufficient
    return state

def llm_select_relevant_links_node(state):
    links = state["links"][:30]
    prompt = (
        f"Question: {state['question']}\n"
        "Here are available links from the page:\n" +
        "\n".join([f"- {l['text']} ({l['href']})" for l in links]) +
        "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        "Reply with a JSON array of up to 3 objects with 'text' and 'href'."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    output = result.content.strip()
    json_str = extract_json_from_text(output)
    try:
        if json_str is None:
            raise ValueError("No JSON array found in LLM output")
        selected_links = json.loads(json_str)
        if not isinstance(selected_links, list):
            raise ValueError
        state["selected_links"] = selected_links[:3]
    except Exception:
        state["selected_links"] = []
    return state

def retrieve_and_answer_node(state: SmartHopState) -> SmartHopState:
    s1 = enhance_query_node(state=type("S", (), dict(**state.dict()))())
    s1 = retrieve_node(s1)
    s1 = answer_node(s1)
    state.answer = s1.answer
    state.sources = s1.used_chunks
    return state

def check_sufficiency_node(state: SmartHopState) -> SmartHopState:
    out = answer_sufficiency_llm_node({
        "question": state.question,
        "answer": state.answer
    })
    state.sufficient = out["sufficient"]
    return state

def pick_next_link_node(state: SmartHopState) -> SmartHopState:
    from urllib.parse import urlparse
    original_domain = state.original_domain or urlparse(state.page_url).netloc
    unvisited_links = [
        l for l in state.links
        if l["href"] not in (state.visited_urls or [])
        and urlparse(l["href"]).netloc == original_domain
    ]
    if not unvisited_links:
        state.selected_link = None
        return state
    out = llm_select_relevant_links_node({
        "question": state.question,
        "links": unvisited_links,
    })
    next_links = out.get("selected_links", [])
    state.selected_link = next_links[0] if next_links else None
    return state

def fetch_link_node(state: SmartHopState) -> SmartHopState:
    if not state.selected_link:
        return state
    url = state.selected_link["href"]
    state.visited_urls.append(url)
    try:
        resp = requests.get(url, timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")
        state.text = soup.get_text(separator="\n", strip=True)
        state.page_url = url
        page_links = [
            {"text": a.get_text(strip=True), "href": a.get("href")}
            for a in soup.find_all("a", href=True)
            if a.get("href", "").startswith("http")
        ]
        state.links = [l for l in page_links if l["href"].startswith("http")]
        state.hops += 1
    except Exception:
        state.selected_link = None
    return state

def hops_remaining(state: SmartHopState):
    return state.hops < 5

graph = StateGraph(SmartHopState)
graph.add_node("RetrieveAndAnswer", retrieve_and_answer_node)
graph.add_node("CheckSufficiency", check_sufficiency_node)
graph.add_node("PickNextLink", pick_next_link_node)
graph.add_node("FetchLink", fetch_link_node)
graph.add_edge("RetrieveAndAnswer", "CheckSufficiency")
graph.add_conditional_edges(
    "CheckSufficiency",
    lambda s: "end" if s.sufficient or not hops_remaining(s) or not s.links else "pick",
    {
        "end": END,
        "pick": "PickNextLink",
    },
)
graph.add_conditional_edges(
    "PickNextLink",
    lambda s: "end" if s.selected_link is None else "fetch",
    {
        "end": END,
        "fetch": "FetchLink",
    },
)
graph.add_edge("FetchLink", "RetrieveAndAnswer")
graph.set_entry_point("RetrieveAndAnswer")
smart_qa_graph = graph.compile()
