# Overview and Deployment

 this is the link of the deployment on google I may close it because it cost money <br>
 https://image-captioning-service-ecdn2gcbaq-uc.a.run.app/Image_Captioning

 this is the deployment on streamlit but it has a problem with the caption model because it need memory more than streamlit offer <br>
 https://blip-model-image-captioning.streamlit.app/
 
 I generated captions to images using blip model and deployed the model on google cloud and I used streamlit for visualization <br>
 I also used the Rag concept with langchain and see results using langsmith <br>

 
# RAG-Generative-AI-Chatbot-link

This repository contains a Streamlit-based application that integrates Google Generative AI for building a chatbot. The chatbot is enhanced with RAG (Retrieval-Augmented Generation) capabilities and allows users to give links of websites for semantic similarity search and context retrieval.

## Features

- **Chatbot Interface**: A user-friendly chatbot interface using Streamlit.
- **web link**: give it a link of a website to extract and split text.
- **Vector Database**: Create a chroma vector database from the extracted text.
- **Semantic Similarity Search**: Use Google Generative AI embeddings for semantic similarity search within the website link.
- **Rag Decomposition**: decomposed the question into 2 questions so that the second question learn from the first question
- **semantic Routing**: compare the similarity between query and prompt and choose the suitable prompt
### Running the Application

To start the Streamlit application, run the following command:
```bash
streamlit run welcome_page.py --server.enableXsrfProtection fals
