In this project, a Voice Assistant Application has been built using Streamlit in Python with Langchain, RAG and Llama3.3 model (Open Source LLM). 
pyttsx3 (text-to-speech library), speech_recognition (speech-to-text library), Hugging Face Instruct Embeddings and Chroma (open-source vector database), 
PyPDFLoader etc. Retrieval-augmented generation (RAG) is an AI framework that combines information retrieval system with large language models (LLMs).
In RAG (Retrieval Augmented Generation), text data from pdf is first splitted into multiple chunks. Then It is converted into vector representations 
(by transforming chunks of text data into embeddings) and stored into Vector Database.
The user query is also converted into an embedding and the system checks the query embedding with the stored document embeddings in vector database using similarity search and identifies and retrieves the most relevant chunks whose embeddings most closely match with the query embedding. Then the retrieved text chunk data along with 
the user query are fed into LLM (in our case it is Open source Llama3.3 model). LLM generates a suitable answer to the user’s query.
 
After launching the Streamlit App, user only needs to upload one document (pdf).
Then they just need to click on "Start Voice Chat" button and our Voice assistant called "Evi"
will start chatting with the user regarding the document. 
It will be a Real time voice chat with the user based on the content of the document.
User can ask any questions about the content of the document and Evi will answer that one after another.
