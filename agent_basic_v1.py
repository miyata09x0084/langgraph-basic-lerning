import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
try:
    from langgraph.types import interrupt
except ImportError:
    from langgraph.utils import interrupt
from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.prebuilt import ToolNode, tools_condition

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

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    # This is a placeholder - actual interrupt happens in dedicated node
    return f"Human assistance requested for: {query}"

# Initialize tools (after API keys are set)
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

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

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Create memory saver for conversation history
memory = InMemorySaver()

# Compile the graph with checkpointer
graph = graph_builder.compile(checkpointer=memory)


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
def stream_graph_updates(user_input: str, config: dict):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config,
        stream_mode="values",
    ):
        # Only print non-empty assistant messages
        content = event["messages"][-1].content
        if content:
            print("Assistant:", content)

# Main interaction loop - following official tutorial
if __name__ == "__main__":
    # Step 4: Prompt the chatbot
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}
    
    print("=== Initial Request ===")
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Check if execution was interrupted
    snapshot = graph.get_state(config)
    print(f"\nNext node: {snapshot.next}")
    
    if snapshot.interrupts:
        print("\n=== Human Assistance Requested ===")
        for interrupt in snapshot.interrupts:
            if isinstance(interrupt.value, dict) and 'query' in interrupt.value:
                print(f"Query: {interrupt.value['query']}")
        
        # Step 5: Resume execution
        print("\n=== Resuming with Expert Response ===")
        human_response = (
            "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
            " It's much more reliable and extensible than simple autonomous agents."
        )
        
        human_command = Command(resume={"data": human_response})
        
        events = graph.stream(human_command, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()




