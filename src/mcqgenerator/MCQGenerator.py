import os
import json
import pandas as pd
import traceback
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import PyPDF2
import anyio
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import get_table_data,read_file
#from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

load_dotenv()

key=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(api_key=key,model_name='gpt-3.5-turbo',temperature=.3)

TEMPLATE='''
Text:{text}
You are an expert MCQ maker. Given the above text it is your job to create quiz of {number} multiple choice questions for {subject} students 
in {tone} tone. Make sure the questions are not repeated and check all questions to be conforming the text as well. make sure the RESPONSE_JSON format is followed.
make sure to format your response like RESPONSE_JSON and use it as a guide. \
ensure to make {number} of MCQs

###RESPONSE_JSON
{response_json}
'''
quiz_generation_prompt=PromptTemplate(
    input_variables=['text','number','subject','tone','response_json'],
    template=TEMPLATE
                     )

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)

TEMPLATE2=''' You are a english grammar expert and writer. Given a MCQ for {subject} students. \
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use max 50 word for complexity, if the 
quiz is not as per the cognitive and analytical ability of the students, \
update the quiz questions which needs to be changed and change the tone that it perfectly fit to the students ability. make sure the RESPONSE_JSON format is followed.
Quiz MCQ: 
{quiz}

Check from an expert english writer of the above quiz
'''
quiz_evaluation_prompt= PromptTemplate(input_variables=['subject','quiz'],template=TEMPLATE2)

review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key='review',verbose=True)

#generate sequential chain
generate_evaluate_chain= SequentialChain(chains=[quiz_chain,review_chain],input_variables=['text','number','subject','tone','response_json'],
                                         output_variables=['quiz','review'],verbose=True)