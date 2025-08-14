from google import genai
from google.genai import types

def main():
    client = genai.Client(
        vertexai=True,
        project="gen-lang-client-0545494042",
        location="global",
    )

    model = "gemini-2.5-flash-lite"

    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus="projects/gen-lang-client-0545494042/locations/us-central1/ragCorpora/4611686018427387904"
                        )
                    ],
                    similarity_top_k=20,
                )
            )
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        tools=tools,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    print("Interactive Vertex AI RAG Console")
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
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                print(chunk.text, end="")
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
