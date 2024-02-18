import os
import json
import pandas as pd
import traceback
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import get_table_data,read_file
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

#loading json file
with open('C:\\Users\\HARI\\mcq_generator\\Response.json','r') as file:
    RESPONSE_JSON=json.load(file)

#create title for app
st.title('MCQ Creator app')

#create form using st.form
with st.form('user_inputs'):
    #file uploader
    uploaded_file=st.file_uploader('Upload a PDF or text file')

    #input varables
    mcq_count=st.number_input('No. of MCQ',min_value=3,max_value=50)
    subject=st.text_input('Insert Subject',max_chars=30)
    tone=st.text_input('Complexity of level of questions',max_chars=20,placeholder='simple')

    #add button
    button=st.form_submit_button('Create MCQ')

if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner('loading....'):
        try:
            text = read_file(uploaded_file)
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        'text': text,
                        'number': mcq_count,
                        'subject': subject,
                        'tone': tone,
                        'response_json': json.dumps(RESPONSE_JSON)
                    }
                )

        except Exception as e:
            st.exception(e)
            st.error('Error')

        else:
            # Use st.write or st.text instead of print
            st.write(f'Total Tokens: {cb.total_tokens}')
            st.write(f'Prompt Tokens: {cb.prompt_tokens}')
            st.write(f'Completion Tokens: {cb.completion_tokens}')
            st.write(f'Total Cost: {cb.total_cost}')

            if isinstance(response, dict):
                quiz = response.get('quiz', None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)
                        st.text_area(label='Review', value=response['review'])
                    else:
                        st.error('Error in table data')

                else:
                    st.write(response)


       
