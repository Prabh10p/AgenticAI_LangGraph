from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

# Load environment
load_dotenv()

# Step 1 - Define LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

# Step 2 - Define State
class LLMBlog(TypedDict):
    title: str
    outline: str
    blog: str
    score: str

# Step 3 - Create Graph
graph = StateGraph(LLMBlog)

# Step 4 - Define Node Functions
def create_outline(state: LLMBlog):
    title = state["title"]
    prompt = f"Create a proper outline on the topic: {title}"
    outline = model.invoke(prompt).content
    return {"outline": outline}   # ✅ return dict, not full state


def create_blog(state: LLMBlog):
    title = state["title"]
    outline = state["outline"]
    prompt = f"Create a detailed blog on '{title}' using the following outline:\n{outline}"
    blog = model.invoke(prompt).content
    return {"blog": blog}   # ✅ return dict


def give_rating(state: LLMBlog):
    title = state["title"]
    blog = state["blog"]
    prompt = f"Evaluate the following blog on '{title}' and rate it out of 10 for clarity and precision:\n\n{blog}"
    score = model.invoke(prompt).content
    return {"score": score}   # ✅ return dict

# Step 5 - Add Nodes and Edges
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_node("give_rating", give_rating)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", "give_rating")
graph.add_edge("give_rating", END)

# Step 6 - Compile Graph
workflow = graph.compile()

# Step 7 - Run
initial_state = {"title": "Tell me About Machine Learning"}
result = workflow.invoke(initial_state)

print(result["score"])
