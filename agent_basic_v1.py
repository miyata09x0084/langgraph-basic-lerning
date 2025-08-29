import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
from tool_node import BasicToolNode

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

# Initialize tools (after API keys are set)
tool = TavilySearch(max_results=2)
tools = [tool]

# Initialize the LLM
llm = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
       ai_message = state[-1]
    elif messages := state.get("messages"):
      ai_message = messages[-1]
    else:
      raise ValueError("No messages found in state")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    else:
        return END

# Build the graph
graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
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




