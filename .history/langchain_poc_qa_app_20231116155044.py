__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
# https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/11

import pdb
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import  ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate


# A ConversationalRetrievalChain is similar to a RetrievalQAChain, except that the ConversationalRetrievalChain allows for passing in of a chat history which can be used to allow for follow up questions.
from langchain.chains import ConversationalRetrievalChain

def identify_source_language(query):
    system_template_lang_detection = "I want you act as a language detector. I will input a sentence in any language and you will answer me in which language the sentence I wrote is in. Do not write any explanations or other words, just reply with the language name"
    human_template_lang_detection = "Sentence: {source_sentence}"

    system_prompt_lang_detection = SystemMessagePromptTemplate.from_template(system_template_lang_detection)
    human_prompt_lang_detection = HumanMessagePromptTemplate.from_template(human_template_lang_detection)
    lang_detection_prompt = ChatPromptTemplate.from_messages([system_prompt_lang_detection, human_prompt_lang_detection])

    lang_detection_chain = LLMChain(llm=llm, prompt=lang_detection_prompt)
    source_language = lang_detection_chain.run(source_sentence=query)
    return source_language


def translate_text(query, source_language, target_language):
    system_template_translation = "You are a good Translator. Translate this sentence from {source_language} to {target_language}. Do not write any other words, just reply with the translated sentence."
    human_template_translation = "Sentence: {source_sentence}"

    system_prompt_translation = SystemMessagePromptTemplate.from_template(system_template_translation)
    human_prompt_translation = HumanMessagePromptTemplate.from_template(human_template_translation)
    translation_prompt = ChatPromptTemplate.from_messages([system_prompt_translation, human_prompt_translation])
    translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
    target_sentence = translation_chain.run(source_language=source_language,source_sentence=query,target_language=target_language)
    return target_sentence


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file, encoding='utf8')
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer

def ask_and_get_answer_only_if_present(vector_store, q, k=3):
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    answer = chain.run(q)
    return answer

def ask_with_memory(vector_store, question, chat_history=[], k=3):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    print('question: ',question)
    print('chat_history: ',chat_history)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    print('chat_history new : ',chat_history)

    return result['answer'], chat_history


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":

    load_dotenv(find_dotenv(), override=True)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    database_language='english'

    st.image('lg.jpg',width=300)
    st.subheader('LG Chatbot')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    lang_detection = st.checkbox('Enable Language Detection',key="no_refresh_checkbox")


    # user's question text input widget
    q = st.text_input('User Query:')
    chat_history=[]
    if q: # if the user entered a question and hit enter

        cur_question=q
        if lang_detection:
            source_language = identify_source_language(q)
            st.text_area('Source Language: ', value=source_language)

            if source_language.lower()!=database_language:
                q=translate_text(q, source_language, database_language)
                st.text_area('Translated Query: ', value=q)


        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)

            vector_store = st.session_state.vs
            #st.write(f'k: {k}')
            #answer = ask_and_get_answer(vector_store, q, k)
            answer, chat_history = ask_with_memory(vector_store, q, chat_history=chat_history, k=k)
            #answer = ask_and_get_answer_only_if_present(vector_store, q, k)

            #chat_history.append((q, answer))

            print('Answer: ',answer)
            print('chat_history final',chat_history)

            # text area widget for the LLM answer
            st.text_area('Response: ', value=answer)

            if lang_detection:
                if source_language.lower()!=database_language:
                    answer=translate_text(answer,database_language, source_language)
                    st.text_area('Translated Response: ', value=answer)


            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {cur_question} \nA: {answer}'
            # print(value)

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
