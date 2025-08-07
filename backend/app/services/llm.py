from langchain_openai import ChatOpenAI
from app.config import config
import json
from app.utils.text import extract_json_from_text

openai_api_key = config.OPENAI_API_KEY

def llm_select_relevant_links(question, links, k=3):
    prompt = (
        f"The user is on the website {links[0].get('href', '')} and their question is: {question}\n"
        "These are all the links on the page:\n" +
        "\n".join([f"- {l.get('text','').strip()[:80]} ({l.get('href')})" for l in links]) +
        "\n\nWhich of these links are most likely to contain the answer or helpful information? "
        "Use your general knowledge and the context of the question, the website, or similarity, intuition to select the most relevant links.\n"
        "Only select links that you are 90% confident to contain an answer, or links to lead to the answer."
        "If you are not at least 90% confident that a link contains the answer, do not include it."
        "Do not include links that are not relevant to the question, or that you are not confident about.\n"
        "Reply with a JSON array of up to 0-5 objects (max 5, min 0) with 'text' and 'href'."
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)
    result = llm.invoke([{"role": "user", "content": prompt}])
    output = (result.content or "").strip()
    json_str = extract_json_from_text(output)
    data = json.loads(json_str) if json_str else []
    if isinstance(data, list):
        return data[:k]
    return []