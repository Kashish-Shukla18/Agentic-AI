from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
)
model = ChatHuggingFace(llm=endpoint)

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"question": "What is the capital of France?"})
print(result)