from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LANGSMITH-DEMO1"
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation"
)
model = ChatHuggingFace(llm=endpoint)
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
config={
    'tags':['langchain','langsmith','sequential chain'],
    'metadata':{'source':'langchain','version':'0.1.0'}
}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)