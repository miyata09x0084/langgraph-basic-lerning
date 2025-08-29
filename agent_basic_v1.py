import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
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

def get_required_api_key(key_name: str) -> str:
    """Get required API key from environment variables."""
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"Please set {key_name} in your .env file")
    return api_key

# Get all required API keys
api_key = get_required_api_key("OPENAI_API_KEY")
tavily_api_key = get_required_api_key("TAVILY_API_KEY")

# Set environment variable for libraries that need it
os.environ["OPENAI_API_KEY"] = api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Initialize the LLM
llm = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Generate and save graph visualization
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

# Define stream function for graph updates
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)

# Main interaction loop
if __name__ == "__main__":
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


