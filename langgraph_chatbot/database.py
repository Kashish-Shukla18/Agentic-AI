from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated
import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import SQLiteSaver

load_dotenv()

endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
    )
model=ChatHuggingFace(llm=endpoint)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chatNode(state:ChatState):
    messages=state['messages']
    response=model.invoke(messages)
    return {"messages":response}

import sqlite3
conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)

checkpointer=SQLiteSaver(conn=conn)

graph=StateGraph(ChatState)

graph.add_node("chatNode",chatNode)
graph.add_edge(START,"chatNode")
graph.add_edge("chatNode",END)

chatbot=graph.compile(checkpointer=checkpointer)
