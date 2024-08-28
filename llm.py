import os
import json

import boto3
from langchain_aws import BedrockLLM
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


openai_api_key = os.getenv('OPENAI_API_KEY')
model_choice = os.getenv('MODEL_CHOICE')
db_path = os.getenv('DB_PATH')


def get_llm():
    if not model_choice or model_choice == 'gpt4':
        print('llm = gpt4')
        llm = ChatOpenAI(
            model='gpt-4-turbo', temperature=0.7, api_key=openai_api_key)    
    elif model_choice == 'gpt3':
        print('llm = gpt3')
        llm = OpenAI(
            model='gpt-3.5-turbo-instruct', temperature=0.7, api_key=openai_api_key)
    elif model_choice == 'claude':
        print('llm = claude2')
        session = boto3.Session()
        bedrock_client = session.client(
            service_name='bedrock',
            region_name='us-west-2',
        )
        # output_text = bedrock.list_foundation_models()
        # print(json.dumps(output_text, indent=4))

        llm = BedrockLLM(
            region_name='us-west-2',
            model_id='anthropic.claude-v2',
            model_kwargs={
            'max_tokens_to_sample': 512,
            'temperature': 0.0,
        })
    return llm


def llm_response(llm, input_text):
    llm_res = llm.invoke(input_text)
    print(llm_res)
    return llm_res.content if isinstance(llm, ChatOpenAI) else llm_res


def query_sql(query_text):
    llm = get_llm()
    sqlite_db_path = db_path
    db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")
    chain = SQLDatabaseChain.from_llm(llm, db, return_direct = True)
    llm_res = chain.invoke(query_text)
    print(llm_res)
