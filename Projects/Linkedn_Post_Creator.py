# Iterative Workflow
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

# Step 1 - Define LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

# Step 2 - Define Iteration State
class IterState(TypedDict):
    title: str
    post: str
    evaluation: Literal["approved", "not approved"]
    feedback: str
    iteration: int
    max_iteration: int


# Step 3 - Generate Post
def generate(state: IterState):
    prompt = [
        SystemMessage(content="You are an expert LinkedIn Post Creator."),
        HumanMessage(content=f"Please generate a LinkedIn post on the following topic:\n{state['topic']}")
    ]
    response = model.invoke(prompt)  # Usually returns a string
    return {'post': response}


# Step 4 - Structured Output for Evaluation
class EvaluationOutput(BaseModel):
    evaluation: Literal["approved", "not approved"] = Field(description="Approve or not approve post based on feedback")
    feedback: str = Field(description="Give a brief feedback on the LinkedIn post")


structured_output = model.with_structured_output(EvaluationOutput)


# Step 5 - Evaluate Post
def evaluate(state: IterState):
    prompt = [
        SystemMessage(content="You are an expert Post Evaluator."),
        HumanMessage(
            content=(
                f"Please evaluate the following post:\n\n"
                f"{state['post']}\n\n"
                f"Topic: {state['topic']}\n\n"
                "Generate a brief feedback on this post and also tell if you approve "
                "or not this post based on its quality and content."
            )
        )
    ]
    response = structured_output.invoke(prompt)
    return {
        'feedback': response.feedback,
        'evaluation': response.evaluation
    }


# Step 6 - Decision Function
def decision(state: IterState):
    if state["evaluation"].lower() == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "not approved"


# Step 7 - Optimisation / Iteration Loop
def optimise(state: IterState):
    # Generate prompt for the model
    prompt = [
        SystemMessage(content="You are an expert LinkedIn Post Creator and Evaluator."),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Previous post (if any): {state.get('post', '')}\n\n"
                "Generate a new LinkedIn post on this topic and provide a brief evaluation "
                "with feedback. Also indicate if the post is approved or not."
            )
        )
    ]

    # Get model response
    generated = model.invoke(prompt)
    
    # Update iteration count (+2 as requested)
    state["iteration"] += 1

    return {
        "response": generated,
        "iteration": state["iteration"]
    }




# Step 8 -Defining Nodes
graph = StateGraph(IterState)
graph.add_node('generate',generate)
graph.add_node('evaluate',evaluate)
graph.add_node('optimise',optimise)



graph.add_edge(START,'generate')
graph.add_edge('generate','evaluate')
graph.add_conditional_edge('evaluate',decision,{'approved':END,"not approved":'optimise'})
graph.add_edge('optimise','evaluate')


worfklow   = graph.compile()
initial_state = {
    'topic': "Machine Learning",
    'post': "",
    'iteration': 0,
    'max_iteration': 5
}

print(worfklow.invoke(initial_state))