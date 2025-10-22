from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import TypedDict,Annotated
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

#1 -  Defining a Model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


#2 -  Defining a State
class State(TypedDict):
      user_input:str
      response:str


#3 -  Defining a Node
def generate(state:State):
      def generate(state: State):
    # Get the latest user input from message list
        user_input = state['messages'][-1].content  
        prompt = f"Generate a 500-1000 word essay on the following topic:\n{user_input}"
    
        response = model.invoke(prompt).content
        return {'response':response}


#4 - Create a Node and Edges
graph = StateGraph(State)
graph.add_node('generate',generate)
graph.add_edge(START,'generate')
graph.add_edge('generate',END)

#5 - Compiling Chatbot workflow
chatbot = graph.compile(checkpointer=checkpointer)
#user_input = input("What you want to generate?")
##inital_state ={'user_input':user_input}
#print(chatbot.invoke(inital_state))