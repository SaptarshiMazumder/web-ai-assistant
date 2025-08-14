from google import genai
from google.genai import types

PROJECT_ID = "gen-lang-client-0545494042"
LOCATION   = "global"

RAG_CORPUS = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"

MODEL_NAME = "gemini-2.5-pro"   # stronger reasoning
SIMILARITY_TOP_K = 50           # broader recall
THINK_BUDGET = 1024             # allow planning / multi-retrieval

SYSTEM = (
  "You are an iterative research assistant. "
  "ALWAYS retrieve before answering; call retrieval multiple times if needed. "
  "When queries are vague or abstract, internally infer short sub-queries, then retrieve again. "
  "Synthesize a precise answer grounded in retrieved content and include 2–6 concise bullet sources (with URLs)."
)

def maybe_count_hint(q: str) -> str:
    ql = q.lower()
    if ("how many" in ql) or ("number of" in ql) or ("count" in ql):
        return ("If the answer requires counting items, first enumerate them briefly from retrieved evidence, "
                "then compute and state the exact count.\n")
    return ""

def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS)
                    ],
                    similarity_top_k=SIMILARITY_TOP_K,
                )
            )
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET),
        system_instruction=SYSTEM,
    )

    print("Interactive Vertex AI RAG Console (smart)")
    print("Type 'exit', 'quit', or 'q' to leave.\n")

    while True:
        user_input = input("Enter your query > ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        # small hint for precise counting when needed
        hint = maybe_count_hint(user_input)

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=hint + user_input)]
            ),
        ]

        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                if chunk.text:
                    print(chunk.text, end="")
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
