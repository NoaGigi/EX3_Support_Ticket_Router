This repository contains the university assignment "Support Ticket Router" (TaskNest) – a LangGraph graph where an LLM chooses the path. Keep the implementation simple, readable, and easy for a beginner to explain.

* Use Python compatible with Google Colab and LangGraph's `StateGraph`.
* Define shared state with `TypedDict`.
* Exactly one LLM call per ticket, inside the classify node, using `ChatOpenAI(...).with_structured_output(...)` with `temperature=0`. No other LLM calls, loops, tools, or memory.
* Load the OpenAI API key only from Colab Secrets with `userdata.get("OPENAI_API_KEY")`. Never use `getpass`, and never write, print, or commit an API key.
* Keep the model name in one constant so it is easy to change.
* Follow standard Python naming and formatting conventions, with type hints.
* Only test when authorized; don't commit or push unless explicitly asked.
