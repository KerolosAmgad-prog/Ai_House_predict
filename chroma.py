import os
os.environ['HF_HOME'] = 'F:/huggingface_cache'    # Change to drive with space
os.environ['TRANSFORMERS_CACHE'] = 'F:/huggingface_cache/transformers'

from langchain_community.document_loaders import PyPDFLoader # loads all PDFs from directory -good for batch processing
from langchain_text_splitters import RecursiveCharacterTextSplitter # smart Splitting that respects paragraph boundries 
from langchain_huggingface import HuggingFaceEmbeddings    # using a local Embedding model like mixedbread v1
from langchain_chroma import Chroma # vector database light wight 
from uuid import uuid4  # for generating unique ids for each documnet 
from dotenv import load_dotenv 



# Environment Configuration 
load_dotenv()
DATA_PATH= r"F:\Ai graduation project\Rag.pdf"
CHROMA_PATH= r"Chroma_db"

#Embeddings Model initialization 
embeddings_model=HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

#Vector Store Setup
Vector_store =Chroma(
    collection_name ="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

#Documnet Loading
loader=PyPDFLoader(DATA_PATH)
raw_docs=loader.load()

#Text Splitting 
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 300 character 
    chunk_overlap=100, # chunk 2 takes last 100 char from chunk 1 
    length_function=len,
    is_separator_regex=False  
)
chunks=text_spliter.split_documents(raw_docs)

#ID generating 
uuids = [str(uuid4()) for _ in range(len(chunks))]

#Storage
Vector_store.add_documents(documents=chunks,ids=uuids)
print("Vector store created successfully")