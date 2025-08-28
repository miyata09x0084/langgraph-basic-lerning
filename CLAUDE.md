# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph learning project that demonstrates creating a basic ReAct agent with weather tool integration. The agent is configured to respond in Japanese and uses OpenAI's GPT-4o-mini model.

## Development Setup

### Environment Setup
1. Activate virtual environment:
   ```bash
   source venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Application
```bash
python agent_basic.py
```

## Architecture

### Core Components

**agent_basic.py**: Main implementation file containing:
- `WeatherResponse` Pydantic model with Japanese field names for structured output
- `get_weather()` tool function that returns mock weather data
- LangGraph ReAct agent configured with:
  - OpenAI GPT-4o-mini model
  - InMemorySaver checkpointer for conversation history
  - Japanese language response configuration
  - Structured response format using WeatherResponse model

### Key Dependencies
- **langgraph**: Core framework for building stateful agents
- **langchain**: Foundation for LLM integrations
- **langchain-openai**: OpenAI model integration
- **python-dotenv**: Environment variable management

### Agent Configuration
The agent uses:
- Model: `openai:gpt-4o-mini` with temperature=0
- Prompt: Configured to always respond in Japanese
- Memory: InMemorySaver with thread-based conversation tracking
- Tools: Single `get_weather` function for demonstration

## Important Notes

- The project requires an OpenAI API key to function
- The agent is configured to respond exclusively in Japanese
- WeatherResponse fields use Japanese property names (天気状況, 気温, 湿度, etc.)
- Currently uses mock weather data ("It's always sunny in {city}!")