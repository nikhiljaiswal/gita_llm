__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import  ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

class LLM_QA:
    """
    Defines the method to perform QA over given source of data using LLM Models
    """

    def load_document(self,file):
        """
        Loads PDF, DOCX and TXT files as LangChain Documents

        Args:
            file: the input file which needs to be loaded as Langchain document
        Returns:
            data: <langchain loader> a langchain loader object
        """
        _, extension = os.path.splitext(file)

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

    def chunk_data(self,data, chunk_size=256, chunk_overlap=0):
        """
        Splits the data into chunks

        Args:
            data: <langchain loader> a langchain loader object
            chunk_size: <int> integer input which suggest what should be the chunk size
            chunk_overlap: <int> integer determining the chunk overlap size
        Returns:
            chunks: <langchain chunk object> a langchain chunk object
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks

    def create_embeddings(self,chunks):
        """
        Create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store

        Args:
            chunks: <langchain chunk object> a langchain chunk object
        Returns:
            vector_store: <vector store> a vectore store which stores the generated embeddings
        """
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        return vector_store

    def calculate_embedding_cost(self,texts):
        """
        Calculate embedding cost using tiktoken

        Args:
            texts: <str> the input text
        Returns:
            total_tokens: <str> total tokens present in the input text
            embedding cost: <float> the estimated embedding cost
        """
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        return total_tokens, total_tokens / 1000 * 0.0004

    def identify_source_language(self,query):
        """
        Identifies the language of the query

        Args:
            query: <str> the input query
        Returns:
            source_language: <str> language of the query
        """
        system_template_lang_detection = "I want you act as a language detector. I will input a sentence in any language and you will answer me in which language the sentence I wrote is in. Do not write any explanations or other words, just reply with the language name"
        human_template_lang_detection = "Sentence: {source_sentence}"

        system_prompt_lang_detection = SystemMessagePromptTemplate.from_template(system_template_lang_detection)
        human_prompt_lang_detection = HumanMessagePromptTemplate.from_template(human_template_lang_detection)
        lang_detection_prompt = ChatPromptTemplate.from_messages([system_prompt_lang_detection, human_prompt_lang_detection])

        lang_detection_chain = LLMChain(llm=llm, prompt=lang_detection_prompt)
        source_language = lang_detection_chain.run(source_sentence=query)
        return source_language

    def translate_text(self,query,source_language,target_language):
        """
        Translate the given query from source_language to target_language

        Args:
            query: <str> the input query which needs to be translated
            source_language: <str> source language of the query
            target_language: <str> target language of the query
        Returns:
            target_sentence: <str> the translated sentence in target language
        """
        system_template_translation = "You are a good Translator. Translate this sentence from {source_language} to {target_language}. Do not write any other words, just reply with the translated sentence."
        human_template_translation = "Sentence: {source_sentence}"

        system_prompt_translation = SystemMessagePromptTemplate.from_template(system_template_translation)
        human_prompt_translation = HumanMessagePromptTemplate.from_template(human_template_translation)
        translation_prompt = ChatPromptTemplate.from_messages([system_prompt_translation, human_prompt_translation])
        translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
        target_sentence = translation_chain.run(source_language=source_language,source_sentence=query,target_language=target_language)
        return target_sentence

    def ask_and_get_answer(self,vector_store, query, k=3):
        """
        Given a vector store and input query use LLM Chain to return the answer

        Args:
            vector_store: <vector store> a vectore store which stores the embeddings
            query: <str> the input query
            k: <int> integer determining number of documents to be retreived
        Returns:
            answer: <str> the generated answer
        """
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = chain.run(query)
        return answer

    def ask_and_get_answer_only_if_present(self,vector_store, query, k=3):
        """
        Given a vector store and input query use LLM Chain to return the answer with minimization of hallucinations

        Args:
            vector_store: <vector store> a vectore store which stores the embeddings
            query: <str> the input query
            k: <int> integer determining number of documents to be retreived
        Returns:
            answer: <str> the generated answer
        """
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

        answer = chain.run(query)
        return answer

    def ask_with_memory(self,vector_store, question, chat_history=[], k=3):
        """
        Given a vector store, input question and a context history use theLLM Chain to return the answer

        Args:
            vector_store: <vector store> a vectore store which stores the embeddings
            question: <str> the input query
            chat_history:  <list> a list containing the history of previous asked question and returned answers
            k: <int> integer determining number of documents to be retreived
        Returns:
            answer: <str> the generated answer
            chat_history: <list> a list containing the history of previous asked question and returned answers
        """
        from langchain.chains import ConversationalRetrievalChain
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        print('question: ',question)
        print('chat_history: ',chat_history)
        result = crc({'question': question, 'chat_history': chat_history})
        chat_history.append((question, result['answer']))
        return result['answer'], chat_history

    def clear_history(self):
        """
        Clear the chat history from streamlit session state
        """
        if 'history' in st.session_state:
            del st.session_state['history']


if __name__ == "__main__":

    load_dotenv(find_dotenv(), override=True)
    llm_obj  =  LLM_QA()

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    database_language='english'

    st.image('lg.jpg',width=400)
    st.subheader('LG Chatbot')

    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=llm_obj.clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=llm_obj.clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=llm_obj.clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = llm_obj.load_document(file_name)
                chunks = llm_obj.chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = llm_obj.calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = llm_obj.create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    lang_detection = st.checkbox('Enable Language Detection',key="lang_checkbox")

    maintain_context = st.checkbox('Enable Context Storage',key="context_checkbox")

    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = []

    # user's question text input widget
    q = st.text_input('User Query:')

    if q: # if the user entered a question and hit enter

        cur_question=q
        if lang_detection:
            source_language = llm_obj.identify_source_language(q)
            st.text_area('Source Language: ', value=source_language)

            if source_language.lower()!=database_language:
                q=llm_obj.translate_text(q, source_language, database_language)
                st.text_area('Translated Query: ', value=q)


        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)

            vector_store = st.session_state.vs

            if 'chat_context' in st.session_state:
                chat_context= st.session_state.chat_context

            if maintain_context:
                answer, chat_context = llm_obj.ask_with_memory(vector_store, q, chat_history=chat_context, k=k)
                st.session_state.chat_context = chat_context
            else:
                answer = llm_obj.ask_and_get_answer(vector_store, q)

            # text area widget for the LLM answer
            st.text_area('Response: ', value=answer)

            if lang_detection:
                if source_language.lower()!=database_language:
                    answer=llm_obj.translate_text(answer,database_language, source_language)
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