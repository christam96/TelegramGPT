from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TelegramChatFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)

###
# Retrievers
###
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# from langchain.indexes import VectorstoreIndexCreator

SECRET_KEY = "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"

# Instantiate model
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

# # Get relevant documents ordered by relevance score
# docs = retriever.get_relevant_documents(query)

# # Reorder the documents: Less relevant document will be at the middle of the list and more
# # relevant elements at beginning / end.
# reordering = LongContextReorder()
# reordered_docs = reordering.transform_documents(docs)

# # Confirm that the 4 relevant documents are at beginning and end.
# for result in reordered_docs:
#     print('Result:', result)
#     print('\n\n')
