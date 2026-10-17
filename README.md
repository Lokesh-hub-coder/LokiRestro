# Restaurant order assistant

This Python CLI uses a LangGraph workflow with three nodes: `take_order`, `cook`,
and `serve`. Groq is used only to classify an order-related message and extract
its item and quantity; all customer-facing responses and stock decisions are
made by the application.

The example menu and kitchen inventory are in `main.py`. `take_order` is the
only node that reads or updates the menu. A failed or unavailable order can be
replaced up to three times. A serving failure sends the same order back to the
kitchen.

## Run

Python 3.12 or later is required. Install the project dependencies, then set
`GROQ_API_KEY` in the environment (or in a `.env` file in this directory) and
run:

```powershell
uv sync
uv run python main.py
```

The default model is Groq's `openai/gpt-oss-120b`. Edit the `MENU`,
`KITCHEN_INVENTORY`, and `RECIPE_PER_DISH` dictionaries in `main.py` to change
the sample restaurant data.