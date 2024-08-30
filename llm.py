import os
import json

import boto3
from langchain_aws import BedrockLLM
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


openai_api_key = os.getenv('OPENAI_API_KEY')
model_choice = os.getenv('MODEL_CHOICE')
db_path = os.getenv('DB_PATH')


class LLMPilot:
    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if not self._llm:
            if not model_choice or model_choice == 'gpt4':
                print('llm = gpt4')
                self._llm = ChatOpenAI(
                    model='gpt-4-turbo', temperature=0.7, api_key=openai_api_key)
            elif model_choice == 'gpt3':
                print('llm = gpt3')
                self._llm = OpenAI(
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

                self._llm = BedrockLLM(
                    region_name='us-west-2',
                    model_id='anthropic.claude-v2',
                    model_kwargs={
                    'max_tokens_to_sample': 512,
                    'temperature': 0.7,
                })
        return self._llm


    def llm_response(self, input_text):
        llm_res = self.llm.invoke(input_text)
        print(llm_res)
        return llm_res.content if isinstance(self._llm, ChatOpenAI) else llm_res


    def query_db(self, query_text):
        # query_text = f'{query_text} Generate SQL without pre-amble.'
        query_text = f'{query_text}'
        db = SQLDatabase.from_uri(db_path)
        chain = SQLDatabaseChain.from_llm(self.llm, db, return_direct=True, verbose=True)
        llm_res = chain.invoke(query_text)
        print(llm_res)
        return llm_res
