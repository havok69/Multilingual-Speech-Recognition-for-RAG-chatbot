import whisper
import os
import pinecone
import time

from langchain.chat_models import ChatOpenAI
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from langchain.vectorstores import Pinecone

from transformers import pipeline


model = whisper.load_model("medium")  # Load whisper model

# Input Query audio/video file
result_1 = model.transcribe("content/vid_1.mp4", fp16 = False)  # Transcription result
result_2 = model.transcribe("content/vid_1.mp4", task = 'translate', fp16 = False)  # Translation result

print(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} ")
print("Language : ", result_1['language'])
print("Transcription : ", result_1['text'])
print("Translation : ", result_2['text'])


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Keep your OpenAi key here

chat = ChatOpenAI(
    openai_api_key = "YOUR_OPENAI_API_KEY", # Keep your OpenAi key here
    model = 'gpt-3.5-turbo'
)
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]

messages_1 = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]

# now create a new user prompt
prompt = HumanMessage(
    content="What is so special about Llama 2?"  # Unknown topic for chatbot, to be stored in vector database
)
# add to messages
messages_1.append(prompt)

res = chat(messages_1)
print(res.content)  # Output for unknown topics (Usually Hallucinations)

dataset = load_dataset(
    "jamescalam/llama-2-arxiv-papers-chunked",
    split="train"
)

pinecone.init(
    api_key='YOUR_PINECONE_API_KEY',  # Keep your Pinecone API key here
    environment="YOUR_PINECONE_ENVIRONMENT"  # Keep your Pinecone Environment here
)

index_name = 'YOUR_PINECONE_INDEX_NAME'  # Keep your pinecone index name here

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()

data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['chunk'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

index.describe_index_stats()

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = "What is so special about Llama 2?"  # The Translated output is input as Query to the RAG implemented chatbot

vectorstore.similarity_search(query, k=3)

def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# With RAG for previous question
prompt = HumanMessage(
    content=augment_prompt(query)  # Implementing RAG
)
messages_1.append(prompt)

res = chat(messages_1)
print(res.content)   # New answer using RAG


# Custom Query from Input Multilingual Whisper
query = result_2['text']  # The Translated output is input as Query to the RAG implemented chatbot
vectorstore.similarity_search(query, k=3)

prompt = HumanMessage(
    content=augment_prompt(query)  # Implementing RAG
)
messages.append(prompt)

res = chat(messages)
print(res.content)   # New answer using RAG


# Summarizing RAG Chatbot output
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")   # Load Summarization model
article = res.content  # Chatbot output as input to summarizer

summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])   # Summary of chatbot output