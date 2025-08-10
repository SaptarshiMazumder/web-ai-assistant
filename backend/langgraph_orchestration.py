from langgraph.graph import StateGraph, END
from state import AssistantState
from page_qa import webpage_answer_node

# LangGraph setup — just one node now!
qa_builder = StateGraph(AssistantState)
qa_builder.add_node("Answer", webpage_answer_node)
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("Answer")
qa_graph = qa_builder.compile()