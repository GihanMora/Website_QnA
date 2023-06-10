import streamlit as st
# import langchain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain import OpenAI, VectorDBQA
# from langchain.chains import RetrievalQAWithSourcesChain
# import PyPDF2
from advertools import crawl
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings


  
st.set_page_config(layout="centered", page_title="Website QnA")
st.header("Website QnA")
st.write("---")
  
  
st.header("Enter Website URL")
site = st.text_input("Enter your URL here")


if site is None:
  st.info(f"""Enter Website to Build QnA Bot""")
elif site:
#   st.write(str(site) + " starting to crawl..")
  try:
    with st.spinner(str(site) + " starting to crawl.."):
      crawl(site, 'simp.jl', follow_links=False)
      crawl_df = pd.read_json('simp.jl', lines=True)
      st.write(len(crawl_df))
      crawl_df = crawl_df[['body_text']]
      st.write(crawl_df)

      #load df to langchain
      loader = DataFrameLoader(crawl_df, page_content_column="body_text")
      docs = loader.load()

      #chunking
      char_text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
      doc_texts = char_text_splitter.split_documents(docs)


      #extract embeddings and build QnA Model
      openAI_embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
      vStore = Chroma.from_documents(doc_texts, openAI_embeddings)

      # Initialize VectorDBQA Chain from LangChain
      #deciding model
      model_name = "gpt-3.5-turbo"
      llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"])
      model = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vStore)

      if crawl_df:
        st.write(str(len(uploaded_files)) + " document(s) loaded..")
        st.header("Ask your data")
        user_q = st.text_area("Enter your questions here")
        if st.button("Get Response"):
          try:
            with st.spinner("Model is working on it..."):
              result = model({"question":user_q}, return_only_outputs=True)
              st.subheader('Your response:')
              st.write(result['answer'])
          except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
   except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, crawling resulted in an error :( Please try again with a different URL.')
