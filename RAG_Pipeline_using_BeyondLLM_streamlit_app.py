import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
import os

# Use Streamlit secrets for API key
if 'GOOGLE_API_KEY' in st.secrets:
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
else:
    st.error("GOOGLE_API_KEY not found in secrets. Please add it to your app settings.")
    st.stop()

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

@st.cache_resource
def initialize_pipeline():
    try:
        data = source.fit("https://en.wikipedia.org/wiki/Niger", dtype="url", chunk_size=1024, chunk_overlap=50)
        embed_model = embeddings.HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        retriever = retrieve.auto_retriever(
            data=data,
            embed_model=embed_model,
            type="cross-rerank",
            mode="OR",
            top_k=2
        )
        llm = llms.GeminiModel(
            model_name="gemini-pro",
            google_api_key=os.environ['GOOGLE_API_KEY']
        )
        system_prompt = """
        <s>[INST]
        You are an AI Assistant.
        Keep the answer up to 5 lines unless the user asks for more information.
        [/INST]
        </s>
        """
        pipeline = generator.Generate(
            question="",
            retriever=retriever,
            system_prompt=system_prompt,
            llm=llm
        )
        return pipeline
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return None

# Streamlit app
st.title("RAG Pipeline using BeyondLLM")

# Initialize the pipeline if not already done
if st.session_state['pipeline'] is None:
    with st.spinner("Initializing RAG pipeline..."):
        st.session_state['pipeline'] = initialize_pipeline()
    if st.session_state['pipeline'] is not None:
        st.success("RAG pipeline initialized!")
    else:
        st.error("Failed to initialize RAG pipeline. Please check your configuration and try again.")
        st.stop()

# Display chat history
for message, response in st.session_state['chat_history']:
    st.text(f"You: {message}")
    st.text(f"Bot: {response}")

# User input
message = st.text_input("You:", key="user_input")

# When user sends a message
if st.button("Send"):
    if message:
        try:
            # Update the pipeline with the new question
            st.session_state['pipeline'].question = message
            
            # Get the response from the pipeline
            with st.spinner("Generating response..."):
                response = st.session_state['pipeline'].call()
            
            # Add to chat history
            st.session_state['chat_history'].append((message, response))
            
            # Display the new message and response
            st.text(f"You: {message}")
            st.text(f"Bot: {response}")
            
            # Clear the input
            st.session_state['user_input'] = ""
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Option to clear chat history
if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

# Display RAG Triad Evaluations
if st.button("Show RAG Triad Evaluations"):
    try:
        evals = st.session_state['pipeline'].get_rag_triad_evals()
        st.json(evals)
    except Exception as e:
        st.error(f"Error getting RAG Triad Evaluations: {str(e)}")
