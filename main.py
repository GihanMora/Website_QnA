import streamlit as st
# import langchain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain import OpenAI, VectorDBQA
# from langchain.chains import RetrievalQAWithSourcesChain
# import PyPDF2
from advertools import crawl
import pandas as pd


  
st.set_page_config(layout="centered", page_title="Website QnA")
st.header("Website QnA")
st.write("---")
  
  
st.header("Enter Website URL")
site = st.text_input("Enter your URL here")


if site is None:
  st.info(f"""Enter Website to Build QnA Bot""")
elif site:
  st.write(str(site) + " starting to crawl..")
  crawl(site, 'simp.jl', follow_links=True)
  crawl_df = pd.read_json('simp.jl', lines=True)
  st.write(len(crawl_df))
  crawl_df = crawl_df[['body_text','header_links_text','og:title','h1', 'h2', 'h3', 'h4','h5','title']]

  st.write(crawl_df)




# #file uploader
# uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
# st.write("---")

# if uploaded_files is None:
#   st.info(f"""Upload files to analyse""")
# elif uploaded_files:
#   st.write(str(len(uploaded_files)) + " document(s) loaded..")
  
#   textify_output = read_and_textify(uploaded_files)
  
#   documents = textify_output[0]
#   sources = textify_output[1]
  
#   #extract embeddings
#   embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
#   #vstore with metadata. Here we will store page numbers.
#   vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
#   #deciding model
#   model_name = "gpt-3.5-turbo"
#   # model_name = "gpt-4"

#   retriever = vStore.as_retriever()
#   retriever.search_kwargs = {'k':2}

#   #initiate model
#   llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
#   model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
#   st.header("Ask your data")
#   user_q = st.text_area("Enter your questions here")
  
#   if st.button("Get Response"):
#     try:
#       with st.spinner("Model is working on it..."):
#         result = model({"question":user_q}, return_only_outputs=True)
#         st.subheader('Your response:')
#         st.write(result['answer'])
#         st.subheader('Source pages:')
#         st.write(result['sources'])
#     except Exception as e:
#       st.error(f"An error occurred: {e}")
#       st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
      
        
    
  
  
  
  
  
  
  
  
  
