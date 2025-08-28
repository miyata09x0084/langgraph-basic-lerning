import os
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

os.environ["OPENAI_API_KEY"] = api_key

llm = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

try:
    # Generate PNG image data
    png_data = graph.get_graph().draw_mermaid_png()
    
    # Save to file
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved as graph.png - you can open this file to view the graph!")
    
    # Try to display in Jupyter environment
    try:
        display(Image(png_data))
        print("Graph visualization displayed successfully!")
    except:
        pass  # Not in Jupyter environment
        
except Exception as e:
    print(f"Failed to generate graph PNG: {e}")
    print("\nTrying to display as Mermaid text instead:")
    try:
        print(graph.get_graph().draw_mermaid())
    except:
        print("Could not display graph. Please install: pip install pygraphviz pillow")

def stream_graph_updates(user_input: str):
  for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
    for value in event.values():
      print("Assistant: ", value["messages"][-1].content)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

while True:
  try:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "q"]:
      print("Goodbye!")
      break
    stream_graph_updates(user_input)
  except:
    # fallback if input() is not available
    user_input = "What do you know about langgraph?"
    print("User: " + user_input)
    stream_graph_updates(user_input)
    break


