from langgraph.graph import StateGraph, END
from page_qa import State, answer_node

# LangGraph setup — just one node now!
qa_builder = StateGraph(State)
qa_builder.add_node("Answer", answer_node)
qa_builder.add_edge("Answer", END)
qa_builder.set_entry_point("Answer")
qa_graph = qa_builder.compile()