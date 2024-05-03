##
##
# Based on https://github.com/jon-strabala/easy-webrag-langchain-demo
# By Jon Strabala
#

import argparse
import tempfile
from langchain_community.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from cbcmgr.cb_operation_s import CBOperation
from cbcmgr.exceptions import NotAuthorized
from cbragdemo.demo_prep import cluster_prep
from cbragdemo.demo_reset import cluster_reset


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the _setup and secrets.toml file"
        )
        st.stop()


def save_to_vector_store(uploaded_file, vector_store):
    """Chunk the PDF & store it in Couchbase Vector Store"""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )

        doc_pages = text_splitter.split_documents(docs)

        vector_store.add_documents(doc_pages)
        st.info(f"PDF loaded into vector store in {len(doc_pages)} documents")


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    _embedding,
    index_name,
):
    """Return the Couchbase vector store"""
    vector_store = CouchbaseVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        index_name=index_name,
    )
    return vector_store


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions
    from datetime import timedelta

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = "couchbases://"+connection_string+"/?ssl=no_verify"
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@st.experimental_dialog("Configuring cluster")
def config_cluster(hostname, username, password, bucket, scope, collection, index_name, project_name, database_name, capella_api_key):
    st.write(f"Configuring cluster")
    try:
        cluster_prep(hostname, username, password, bucket, scope, collection, index_name, project=project_name, database=database_name, api_key=capella_api_key)
    except Exception as e:
        st.write(f"Error: {e}")
    else:
        st.write(f"Success")
    if st.button("Ok"):
        st.rerun()


@st.experimental_dialog("Resetting cluster")
def reset_cluster(hostname, username, password, bucket, scope, collection):
    st.write(f"Flushing bucket")
    try:
        cluster_reset(hostname, username, password, bucket, scope, collection)
    except Exception as e:
        st.write(f"Error: {e}")
    else:
        st.write(f"Success")
    if st.button("Ok"):
        st.rerun()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-u', '--user', action='store', help="User Name", default="Administrator")
    parser.add_argument('-p', '--password', action='store', help="User Password", default="password")
    parser.add_argument('-h', '--host', action='store', help="Cluster Node Name", default="localhost")
    parser.add_argument('-b', '--bucket', action='store', help="Bucket", default="vectordemos")
    parser.add_argument('-s', '--scope', action='store', help="Scope", default="langchain")
    parser.add_argument('-c', '--collection', action='store', help="Collection", default="webrag")
    parser.add_argument('-i', '--index', action='store', help="Index Name", default="webrag_index")
    parser.add_argument('-P', '--project', action='store', help="Project Name")
    parser.add_argument('-D', '--database', action='store', help="Capella Database")
    parser.add_argument('-R', '--profile', action='store', help="Capella API Profile", default="default")
    options = parser.parse_args()
    return options


def main():
    # Authorization
    if "auth" not in st.session_state:
        st.session_state.auth = False

    st.set_page_config(
        page_title="Chat with your PDF using Langchain, Couchbase & OpenAI",
        page_icon="ð¤",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    options = parse_args()

    AUTH = options.password

    # Authentication
    host_name = st.text_input("Couchbase Server Hostname", "127.0.0.1")
    user_name = st.text_input("Username", "Administrator")
    user_password = st.text_input("Enter password", type="password")
    open_api_key = st.text_input("Enter OpenAPI Key", type="password")
    bucket_name = st.text_input("Bucket", "vectordemos")
    scope_name = st.text_input("Scope", "langchain")
    collection_name = st.text_input("Collection", "webrag")
    index_name = st.text_input("Index Name", "webrag_index")
    project_name = st.text_input("Capella Project Name")
    database_name = st.text_input("Database Name")
    capella_api_key = st.text_input("Capella API Key", type="password")
    pwd_submit = st.button("Submit")
    config_button = st.button("Configure")
    reset_button = st.button("Reset")

    if config_button:
        config_cluster(host_name, user_name, user_password, bucket_name, scope_name, collection_name, index_name, project_name, database_name, capella_api_key)

    if reset_button:
        reset_cluster(host_name, user_name, user_password, bucket_name, scope_name, collection_name)

    if pwd_submit:
        try:
            CBOperation(host_name, user_name, user_password, ssl=True)
        except NotAuthorized:
            st.error("Incorrect password")
        else:
            st.session_state.auth = True

    os.environ["OPENAI_API_KEY"] = open_api_key

    if st.session_state.auth:
        # Set Couchbase variables
        CB_HOSTNAME = host_name
        CB_USERNAME = user_name
        CB_PASSWORD = user_password
        CB_BUCKET = bucket_name
        CB_SCOPE = scope_name
        CB_COLLECTION = collection_name
        CB_SEARCHINDEX = index_name

        # Use OpenAI Embeddings
        embedding = OpenAIEmbeddings()

        # Connect to Couchbase Vector Store
        cluster = connect_to_couchbase(CB_HOSTNAME, CB_USERNAME, CB_PASSWORD)

        vector_store = get_vector_store(
            cluster,
            CB_BUCKET,
            CB_SCOPE,
            CB_COLLECTION,
            embedding,
            CB_SEARCHINDEX,
        )

        # Use couchbase vector store as a retriever for RAG
        retriever = vector_store.as_retriever()

        # Build the prompt for the RAG
        template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        # Use OpenAI GPT 4 as the LLM for the RAG
        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

        # RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Pure OpenAI output without RAG
        template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

        Question: {question}"""

        prompt_without_rag = ChatPromptTemplate.from_template(template_without_rag)

        llm_without_rag = ChatOpenAI(model="gpt-4-1106-preview")

        chain_without_rag = (
            {"question": RunnablePassthrough()}
            | prompt_without_rag
            | llm_without_rag
            | StrOutputParser()
        )

        # Frontend
        couchbase_logo = (
            "https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/couchbase.png"
        )
        openai_logo = (
            "https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/openapi.png"
        )

        st.title("Chat with PDF")
        st.markdown(
            "Answers with ![Couchbase logo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/couchbase.png) are generated using *RAG* while ![OpenAI logo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/openapi.png) are generated by pure *LLM (ChatGPT)*"
        )

        with st.sidebar:
            st.header("Upload your PDF")
            with st.form("upload pdf"):
                uploaded_file = st.file_uploader(
                    "Choose a PDF.",
                    help="The document will be deleted after one hour of inactivity (TTL).",
                    type="pdf",
                )
                submitted = st.form_submit_button("Upload & Vectorize")
                if submitted:
                    # store the PDF in the vector store after chunking
                    save_to_vector_store(uploaded_file, vector_store)

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you will get two answers:
                * one using RAG ![Couchbase logo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/couchbase.png)
                * one using pure LLM - OpenAI ![OpenAI logo](https://raw.githubusercontent.com/mminichino/cb-rag-langchain-demo/main/doc/openapi.png).
                """
            )

            st.markdown(
                "For RAG, we are using [Langchain](https://langchain.com/), [Couchbase Vector Search](https://couchbase.com/) & [OpenAI](https://openai.com/). We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store."
            )

            # View Code
            if st.checkbox("View Code"):
                st.write(
                    "View the code here: [Github](https://github.com/mminichino/cb-rag-langchain-demo/blob/main/cbragdemo/chat_with_pdf.py)"
                )

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                    "avatar": openai_logo,
                }
            )

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        # React to user input
        if question := st.chat_input("Ask a question based on the PDF(s)"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": openai_logo}
            )

            # Add placeholder for streaming the response
            with st.chat_message("assistant", avatar=couchbase_logo):
                message_placeholder = st.empty()

            # stream the response from the RAG
            rag_response = ""
            for chunk in chain.stream(question):
                rag_response += chunk
                message_placeholder.markdown(rag_response + "â")

            message_placeholder.markdown(rag_response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    "avatar": couchbase_logo,
                }
            )

            # stream the response from the pure LLM

            # Add placeholder for streaming the response
            with st.chat_message("ai", avatar=openai_logo):
                message_placeholder_pure_llm = st.empty()

            pure_llm_response = ""

            for chunk in chain_without_rag.stream(question):
                pure_llm_response += chunk
                message_placeholder_pure_llm.markdown(pure_llm_response + "â")

            message_placeholder_pure_llm.markdown(pure_llm_response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": pure_llm_response,
                    "avatar": openai_logo,
                }
            )


if __name__ == '__main__':
    main()
