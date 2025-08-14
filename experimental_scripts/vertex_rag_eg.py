from google import genai
from google.genai import types

PROJECT_ID = "gen-lang-client-0545494042"
LOCATION   = "global"

RAG_CORPUS = "projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"

MODEL_NAME = "gemini-2.5-pro"   # stronger reasoning
SIMILARITY_TOP_K = 50           # broader recall
THINK_BUDGET = 1024             # allow planning / multi-retrieval

# Single source of truth: no special-casing.
SYSTEM = """
You are an advanced research and analysis assistant with access to a Vertex AI RAG corpus.
Your goals are to:
1. Understand the user's request fully — even if it is vague, abstract, or multi-part.
2. If needed, break the request into smaller sub-questions internally and retrieve content multiple times.
3. Use retrieved content as primary evidence. Pull relevant, high-value snippets and note their sources.
4. Perform analytical reasoning:
   - Compare and contrast findings
   - Summarize patterns or trends
   - Calculate counts, sums, ratios, percentages, or differences where useful
   - Deduplicate and cluster similar items
   - Handle multi-step reasoning and conditional logic
5. Ensure factual grounding: verify all claims against retrieved evidence.
6. Produce a clear, concise, and logically structured final answer.
7. Include 2–6 concise bullet point citations with URLs at the end, showing key supporting sources.

When you retrieve:
- Target specific, relevant pages or sections.
- Make multiple retrieval calls if the question has more than one focus or dimension.

When you answer:
- Structure your answer so it can be understood by someone unfamiliar with the source material.
- Be explicit about your reasoning process when it adds clarity.
"""


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
        temperature=0,
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

    print("Interactive Vertex AI RAG Console (no heuristics)")
    print("Type 'exit', 'quit', or 'q' to leave.\n")

    while True:
        user_input = input("Enter your query > ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
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
