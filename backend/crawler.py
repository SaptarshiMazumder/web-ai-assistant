import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())