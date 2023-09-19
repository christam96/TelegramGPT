import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TelegramChatFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load OpenAI API key
load_dotenv()
SECRET_KEY = os.environ.get('SECRET_KEY')

# Instantiate LLM model
llm = ChatOpenAI(openai_api_key=SECRET_KEY)

# Load Telegram data
loader = TelegramChatFileLoader("Data/result.json")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,
)
data = loader.load_and_split(text_splitter)
print('Chat history has been split into', len(data), 'documents.')

# Create text embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)
vectorstore = Chroma("telegram_store", embeddings_model)
retriever = Chroma.from_documents(documents=data, 
                                  embedding=embeddings_model).as_retriever(search_kwargs={"k":5})


qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever)
query = "What has been discussed about zkRoute?"
print("Query:", query)
print(qa.run(query))
