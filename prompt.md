# Codex Spec: TaskNest Support Ticket Router

Build a tiny LangGraph graph in which one decision is made by an LLM: the `classify` node
calls the LLM and writes its decision into the state, and a single conditional edge reads
that decision to choose the next node. Build exactly this graph and nothing more.

## 1. Deliverable

A self-contained Google Colab notebook named `ticket_router.ipynb` (standard CPU runtime).

## 2. Model

- Model: `gpt-6-luna` through `ChatOpenAI`, with `temperature=0` and `reasoning_effort='none'`.
- Why: sorting a ticket into one of three teams is an easy task, so one of the cheapest
  OpenAI text models is enough. `temperature=0` makes the same ticket go to the same team.
- Define the model name once, as a constant at the top of the notebook:
  `MODEL_NAME = "gpt-6-luna"`. No other cell may hard-code the model name.

## 3. API key

Load the key from Colab Secrets only:

```python
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

No `getpass`. Never write, print, or output the key.

## 4. State

```python
class TicketState(TypedDict, total=False):
    ticket: str     # input: the original ticket text
    category: Literal["billing", "technical", "general"]  # chosen by the LLM
    team: str       # the team the ticket was forwarded to
    reply: str      # fixed short note to the customer
```

## 5. Structured output

A small Pydantic model whose `category` field is a `Literal` of the three names:

```python
class TicketCategory(BaseModel):
    category: Literal["billing", "technical", "general"]
```

Create the classifier once, outside the node:
`classifier = ChatOpenAI(model=MODEL_NAME, temperature=0, reasoning_effort="none").with_structured_output(TicketCategory)`

## 6. Nodes

### `classify` (the only node that calls the LLM)
- Reads `state["ticket"]`, calls the classifier exactly once, and returns only
  `{"category": ...}`.
- Send this system instruction, with the ticket text as a separate human message:

  > You are a support ticket classifier for TaskNest, a to-do app.
  > Classify the ticket into exactly one category:
  > - billing: payments, charges, invoices, refunds, subscriptions.
  > - technical: bugs, crashes, error messages, login problems.
  > - general: feature ideas, general questions, thank-you notes, anything else.
  > Decide by the customer's main problem, not by single words. If something is broken
  > (for example, the app crashes while paying), it is technical. If money was charged
  > wrongly, it is billing.

### Team nodes (plain Python, no LLM)
Each node sets `team` and a fixed `reply`:

| Node | `team` | `reply` |
|---|---|---|
| `billing_team` | `"Billing"` | `"Thanks for reaching out! Your ticket was forwarded to our Billing team."` |
| `tech_team` | `"Tech"` | `"Thanks for reaching out! Your ticket was forwarded to our Tech team."` |
| `general_team` | `"General"` | `"Thanks for reaching out! Your ticket was forwarded to our General support team."` |

## 7. Graph

```
START → classify
classify --route(state)--> "billing"   → billing_team → END
                           "technical" → tech_team    → END
                           "general"   → general_team → END
```

- Routing function: `route(state)` only reads and returns `state["category"]`. It does not
  call the LLM.
- Exactly one `add_conditional_edges` from `classify`, mapping each category to its team node.
- No extra LLM calls, no loops, no tools, no memory.
- The graph must run as: `result = graph.invoke({"ticket": "I was charged twice this month"})`
  and the result must contain `ticket`, `category`, `team` and `reply`.

## 8. Notebook cells (markdown heading above each)

1. Install: `!pip -q install langgraph langchain-openai`
2. API key from Colab Secrets.
3. Imports and `MODEL_NAME`.
4. State and `TicketCategory`.
5. Classifier setup.
6. `classify` node. Mark the model call line with `# LLM call`.
7. `route` function. Mark the return line with `# LLM decision chooses the edge`.
8. Team nodes.
9. Build and compile the graph.
10. Draw the graph with `draw_mermaid_png()`.
11. One example run with `"I was charged twice this month"`.
12. Test tickets.

## 9. Test tickets

For each ticket, run `graph.invoke` and print the ticket, the expected category, the
category the LLM chose, and the reply. If the LLM's choice differs from the expected one,
mark it clearly but do not stop the notebook.

Always assert: every result has the four keys, and every branch is reached at least once.

| # | Ticket | Expected |
|---|---|---|
| 1 | I was charged twice this month | billing |
| 2 | How do I get a refund for my annual subscription? | billing |
| 3 | The app crashes every time I open a shared list | technical |
| 4 | I can't log in, the password reset email never arrives | technical |
| 5 | Could you add a dark mode? | general |
| 6 | Just wanted to say thank you, TaskNest is great! | general |
| 7 | The app crashes every time I try to pay | technical |
| 8 | I paid for Premium but the Premium features don't show up | billing |
| 9 | Do you offer a student discount? | billing |
| 10 | Your app is useless and I want my money back | billing |
| 11 | I can't log in, so I can't cancel my subscription | technical |

## 10. Revision notes

- Round 1: initial spec.
- Round 2: removed ‘(ambiguous)’ from ticket texts so it is not sent to the LLM; split the system instruction into lines.
- Round 3: added reasoning_effort='none' because gpt-6-luna rejected temperature=0.
