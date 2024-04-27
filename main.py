# Imports
import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
import time
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
import asyncio

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

# -----------------------------------------------------------------

# _____________________ Graph Design _______________________________

# -------------------------------------------------------------------


# Initiating state variable
class AgentRAG(TypedDict):
    file_path: str    
    file_type: str
    data: str
    db_status: bool
    query: str
    answer: str

# -----------------------------------------------------------------

# _____________________ Creating Nodes ______________________________

# -------------------------------------------------------------------


def csv_analysis(state):
    print("Entering for CSV Analysis")
    input_df = pd.read_csv(state['file_path'])
    agent = create_pandas_dataframe_agent(llm=llm,
                                          df=input_df,
                                          verbose=False,
                                          agent_type=AgentType.OPENAI_FUNCTIONS,
                                          )
    answer = agent.invoke({state['query']})
    print(f"Answer: {answer['output']}")
    # return {'answer': answer['output']}


def file_check(state):
    print("Entering to File Check")
    file_path = state['file_path']
    if os.path.exists(file_path):
        _, file_extension = os.path.splitext(file_path)
        # based on file type directly rag or csv_analyzer can be called,
        # but we can't end if incorrect file type is found, hence, 
        # to add conditional edge, returning file path
        if file_extension.lower() == '.pdf':
            return {'file_type': 'pdf'}         
        elif file_extension.lower() == '.csv':
            return {'file_type': 'csv'}
        else:
            return 'end'
    else:
        return 'end'
    

def extract_content(state):
    print("Entering to extract text from the document")
    file_path = state['file_path']
    
    # using Unstructured module to extract content from pdf 
    elements = partition_pdf(filename=file_path)
    extracted_text = ' '.join([str(element) for element in elements])
    return {'data': extracted_text}


def build_rag(state):
    print("Entering to build RAG system and preparing the database")
    data = state['data']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    doc = text_splitter.create_documents([data])
    docs = text_splitter.split_documents(doc)

    embed = OpenAIEmbeddings()

    # Preparing Pinecone vector db
    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    index_dict = index.describe_index_stats()

    if os.environ["NAMESPACE"] in list(index_dict['namespaces'].keys()):
        index.delete(delete_all=True, namespace=os.environ["NAMESPACE"])
    else:
        pass

    print("Now pushing to Pinecone VectorDB")

    Pinecone.from_documents(docs, embed,
                            index_name=os.environ["PINECONE_INDEX_NAME"],
                            namespace=os.environ["NAMESPACE"])
    delay = 30
    print(f"adding delay for {delay} seconds")
    time.sleep(delay)
    
    return {'db_status': True}


def get_answer(state):
    embed = OpenAIEmbeddings()
    index = Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"],
                                         embedding=embed, 
                                         namespace=os.environ["NAMESPACE"])
    query = state['query']

    docs = index.similarity_search(query, k=5)

    chain = load_qa_chain(llm=llm, chain_type='stuff')
    
    response = chain.invoke({'question': query, 'input_documents': docs})

    print(f"Answer: {response['output_text']}")

    return {'answer': response['output_text']}


def collect_query(state):
    print("Entering in loop for user query")
    if (state['file_type'] == 'pdf' and state['db_status']) or state['file_type'] == 'csv':
        return {'query': input("Enter your query... ")}
    

# -----------------------------------------------------------------

# ___________________ Creating Conditional Edges ___________________

# -------------------------------------------------------------------


def decide_agent(state):
    print("Entering to decide for RAG or CSV analysis")
    if state["file_type"] == 'pdf':
        return 'rag'
    else:  # if state["file_type"] == 'csv':
        return 'csv'


def decide_to_end_on_user_query(state):
    print("Entering to decide if user want to end")
    if 'query' in state and state['query'] == 'exit':
        return 'end'
    
    if state['file_type'] == 'pdf':
        return 'get_answer'
    else:
        return 'csv_analysis'
    

# -----------------------------------------------------------------

# _____________________ Workflow configuration _____________________

# -------------------------------------------------------------------


workflow = StateGraph(AgentRAG)

# Define nodes
workflow.add_node('File Check', file_check)
workflow.add_node('Extract Content', extract_content)
workflow.add_node('Build RAG', build_rag)
workflow.add_node('Get Query', collect_query)
workflow.add_node('Get Answer', get_answer)
workflow.add_node('CSV Analyzer', csv_analysis)

# Build graph
workflow.set_entry_point('File Check')
workflow.add_conditional_edges('File Check',
                               decide_agent,
                               {
                                # "end": END,
                                "rag": "Extract Content",
                                "csv": "Get Query",
                               })
workflow.add_edge('Extract Content', 'Build RAG')
workflow.add_edge('Build RAG', 'Get Query')

workflow.add_conditional_edges('Get Query',
                               decide_to_end_on_user_query,
                               {
                                   'end': END,
                                   'get_answer': 'Get Answer',
                                   'csv_analysis': 'CSV Analyzer'
                               })
# workflow.add_edge('Get Answer','Get Query')
workflow.add_edge('Get Answer', 'Get Query')
workflow.add_edge('CSV Analyzer', 'Get Query')

app = workflow.compile()

# -----------------------------------------------------------------

# __________________________ Execution _____________________________

# -------------------------------------------------------------------


# Resume_Ankit.pdf
# customers.csv
inputs = {
    'file_path': 'PATH_TO_YOUR_PDF_OR_CSV_DOCUMENT'
}


config = {"recursion_limit": 7}

app.invoke(inputs, config=config)

# running_dict = {}

# async def process_events():
#     async for event in app.astream(inputs, config=config):
#         for k, v in event.items():
#             running_dict[k] = v
#             if k != "__end__":
#                 print(v)
#                 print('----------' * 20)
#
# asyncio.run(process_events())
