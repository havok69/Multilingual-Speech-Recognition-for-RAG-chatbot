# Multilingual-Speech-Recognition-for-RAG-chatbot
# Documentation


## Building a Multilingual Speech Recognition Model for RAG Without Training

This document details the functionalities, code structure, implementation, and usage guidelines for the pre-trained multilingual speech recognition model, Multilingual Whisper, to enable RAG to perform tasks in multiple languages.




### Setup

#### Environment Setup

Python 3.9.9 and PyTorch 1.10.1 were used to train and test the models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions.

For the entire project, a conda environment was used, which was created using:

```
conda create -n venv python=3.9.18 anaconda
```

The PyTorch version used is PyTorch 1.10.1, which can be installed using:
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
```



#### Whisper dependencies

You can download and install (or update to) the latest release of Whisper with the following command:
```
pip install -U openai-whisper
```
It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
The codebase also depends on a few Python packages, most notably OpenAI's tiktoken for their fast tokenizer implementation. The open-source version of tiktoken can be installed from PyPI:
```
pip install tiktoken
```
You may need rust installed as well, in case tiktoken does not provide a pre-built wheel for your platform. If you see installation errors during the pip install command above, please follow the Getting started page to install the Rust development environment. Additionally, you may need to configure the PATH environment variable, e.g. export PATH="$HOME/.cargo/bin:$PATH". If the installation fails with No module named 'setuptools_rust', you need to install setuptools_rust, e.g. by running:
```
pip install setuptools-rust
```




#### Available models and languages
There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model; actual speed may vary depending on many factors including the available hardware.
The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
For our purpose, we use the `medium` model, which was observed to be the most suitable model, because it is the lightest model with accurate results for transcription as well as translation.




#### Chatbot Dependencies	
Before we start building our chatbot, we need to install some Python libraries. Here's a brief overview of what each library does:

•	`langchain`: This is a library for GenAI. We'll use it to chain together different language models and components for our chatbot.
•	`openai`: This is the official OpenAI Python client. We'll use it to interact with the OpenAI API and generate responses for our chatbot.
•	`datasets`: This library provides a vast array of datasets for machine learning. We'll use it to load our knowledge base for the chatbot.
•	`pinecone-client`: This is the official Pinecone Python client. We'll use it to interact with the Pinecone API and store our chatbot's knowledge base in a vector database.

You can install these libraries using pip like so:
```
pip install langchain==0.0.292 openai==0.28.0 datasets==2.10.1 pinecone-client==2.2.4 tiktoken==0.5.1
```
You'll need to get an OpenAI API key and Pinecone API key to run the chatbot and use a vector database respectively.




#### Dependencies for Transformer-based models for Text Summarization
The command pip install transformers is used to install the transformers package, which provides access to state-of-the-art Transformer-based models for NLP tasks, including Text Summarization.
```
# install transformers
!pip install transformers
```
Once the transformers package is installed, you can import and use the Transformer-based models in your own projects.

Brief Explanation of Code:

•	Loading Models and Libraries

    Imports necessary libraries and models: whisper, os, pinecone, time, various modules from langchain, datasets, and transformers.
  
•	Model Initialization and Usage

    Loads the whisper model and performs transcriptions and translations of an audio/video file.
  
    Whisper model output taken as a query for RAG implemented chatbot.
  
    Initializes a chat model (ChatOpenAI) for interaction and creates messages for the chatbot.
  
    Utilizes the chat model to get responses and prints the content.
  
•	Dataset and Vector Storage

    Loads a dataset related to "llama-2-arxiv-papers-chunked".
  
    Initializes Pinecone for vector storage, creates an index, and updates it with embeddings and metadata.
  
•	Vector Search and Augmented Prompt

    Performs similarity searches in the vector store based on a given query.
  
    Defines a function to create an augmented prompt using retrieved knowledge to assist in answering queries.
  
•	RAG Implementation

    Constructs prompts for the RAG (Retrieval-Augmented Generation) model, appending queries to messages.
  
    Uses the chat model for RAG implementation and prints the new answer using RAG.
  
•	Summarization

    Loads a summarization model.
  
    Summarizes the chatbot output from the RAG using the summarization model and prints the summary.


    

#### Usage Guide
The code, or the file `main.py` can be run using python in the environment previously created, after setting few important parameters, which are:

    audio/video file path
    
    API keys for OpenAI and Pinecone vector db
    
Upon running the code, the outputs of transcription, translation, output for the default query, output for the audio/video transcription, translated and taken as a query and also the summarization for the output of the query, will be output by the code.




#### Summary of the Code Documentation 

    The code incorporates Whisper for speech recognition and OpenAI models for chatbot interactions.

    Utilizes Pinecone for vector storage and retrieval.

    Offers functionality for transcription, translation, and chatbot interactions.

    The code showcases how pre-trained models can be leveraged for various language-related tasks without retraining.







