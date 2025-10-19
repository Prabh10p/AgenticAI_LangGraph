# workflow
# start -> sentiment_analyzer
# if positive -> positive_response -> end
# else -> response_analyzer -> negative_response -> end

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1 Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# 2 Define State
class CurrentState(TypedDict, total=False):
    review: str
    sentiment: Literal['positive','negative']
    comment_response: dict
    response: str

# 3 Define Structured Output Models
class PySentiment(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Give a sentiment of the text")
structured_output = model.with_structured_output(PySentiment)

class PyAnalyzer(BaseModel):
    issue_type: Literal["UX","Performance","Bug","Other"]
    tone: Literal["angry","disappointed","calm"]
    urgency: Literal["low","medium","high"]
structured_output1 = model.with_structured_output(PyAnalyzer)

# 4 Define Functions
def sentiment_analyzer(state: CurrentState):
    prompt = f'Give me a deep sentiment based on the following review:\n{state["review"]}'
    result = structured_output.invoke(prompt)
    return {'sentiment': result.sentiment}

def check_sentiment(state: CurrentState) -> str:
    if state.get("sentiment") == "positive":
        return "positive_response"
    else:
        return "response_analyzer"

def positive_response(state: CurrentState):
    prompt = f'Give positive feedback based on the following review:\n{state["review"]}'
    result = structured_output.invoke(prompt)
    return {'response': result.sentiment}  # You can also make a separate model for positive_response if needed

def response_analyzer(state: CurrentState):
    prompt = f'Analyze issue_type, tone, and urgency for the following review:\n{state["review"]}'
    result = structured_output1.invoke(prompt)
    return {'comment_response': result.dict()}

def negative_response(state: CurrentState):
    cr = state["comment_response"]
    prompt = (
        f'The user has {cr["issue_type"]} issue, '
        f'sounded {cr["tone"]} and marked urgency is {cr["urgency"]}. '
        'Write an empathetic apology message to the user.'
    )
    result = structured_output.invoke(prompt)
    return {'response': result.sentiment}

# 5 Define Graph Nodes
graph = StateGraph(CurrentState)
graph.add_node('sentiment_analyzer', sentiment_analyzer)
graph.add_node('positive_response', positive_response)
graph.add_node('response_analyzer', response_analyzer)
graph.add_node('negative_response', negative_response)

# 6 Define Graph Edges
graph.add_edge(START, 'sentiment_analyzer')
graph.add_conditional_edge('sentiment_analyzer', check_sentiment)
graph.add_edge('positive_response', END)
graph.add_edge('response_analyzer', 'negative_response')
graph.add_edge('negative_response', END)

# 7️ Compile Workflow
workflow = graph.compile()

# 8 Run Example
initial_state = {'review': "The phone is so bad"}
print(workflow.invoke(initial_state))
