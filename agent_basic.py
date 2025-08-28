from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

class WeatherResponse(BaseModel):
    天気状況: str
    気温: float
    湿度: float
    風速: float
    風向: str
    降水量: float
    視界: float
    気圧: float

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

os.environ["OPENAI_API_KEY"] = api_key

checkpointer = InMemorySaver()

model = init_chat_model(
    model="openai:gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="You are a helpful assistant that can answer questions about the weather. Always respond in Japanese. When providing structured data, use Japanese values and descriptions.",
    checkpointer=checkpointer,
    response_format=WeatherResponse
)

config = {"configurable": {"thread_id": "my-unique-thread-id"}}

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        config
    )
    print(response)