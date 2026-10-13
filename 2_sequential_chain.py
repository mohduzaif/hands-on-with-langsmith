from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

# setup an project name on LangSmith portal from the code.
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM Workflow'

# load the environment variables.
load_dotenv()

# generate the prompt1
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# generate the prompt2
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# create the model object.
model = ChatCohere(
    model = 'command-a-03-2025', 
    temperature = 0.7
)

# create the parser.
parser = StrOutputParser()

# create the sequential chain.
chain = prompt1 | model | parser | prompt2 | model | parser

# set the some configuration.
config = {
    'run_name' : 'Sequential Chain', 
    'tags' : ['llm app', 'report generation', 'summarization'], 
    'metadata' : {
        'model' : 'ChatCohere',
        'model_temperature' : 0.7, 
        'parser' : 'stroutputparser'
    }
}

# response from the LLM.
result = chain.invoke(
    {'topic': 'Unemployment in India'}, 
    config = config
)

# print the result.
print(result)
