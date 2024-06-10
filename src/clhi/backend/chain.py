from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
import subprocess
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

PROMPT_TEMPLATE = """You are an assistant for Linux Terminal users. You are answering questions related with terminal commands and your goal is to generate a relevant CLI command the user can run in their CLI.

Make sure to return the generated CLI command as the first line of your response, followed by an explanation of the command and possibly any arguments or option flags you used. Do NOT wrap or surround the generated command in quotes, backticks or other type of code blocks or indications. Make sure the command is one-to-one copy-pastable into a terminal.
You can use the following historical CLI commands of the user related to this question to generate a CLI command to their question:
{context}

Here is the chat history between you and the human: {chat_history}

Question: {question}

If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Answer:
"""


def get_vectorstore_retriever():
    embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
    vsc = VectorSearchClient(disable_notice=True)

    vs_index = vsc.get_index(
        endpoint_name='dais_hackathon',
        index_name='workspace.default.man_7_info_index'
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="Text", embedding=embedding_model
    )

    return vectorstore.as_retriever()

def get_embeddings(user_input: str):
    vectorstore = get_vectorstore_retriever()
    similar_documents = vectorstore.invoke(user_input)
    command_history = [x.metadata["Command"] for x in similar_documents]
    return similar_documents, command_history

def get_command_history(command):
    history = subprocess.Popen(f"cat ~/.zsh_history | grep -m 5 {command}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    history_array = []
    for line in history.stdout.readlines():
        history_array.append(line.decode("utf-8").split(";")[1].strip())
    
    return history_array

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_cli_history(input):
    question = extract_question(input)
    similar_commands = get_embeddings(question)
    output_array = []

    for command in similar_commands:
        output_array.extend(get_command_history(command))
    return output_array

def extract_chat_history(input):
    return input[:-1]

def build_chain():
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 1000)

    chain_with_cli_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "context": itemgetter("messages") | RunnableLambda(extract_cli_history),
    }
    | prompt
    | chat_model
    | StrOutputParser()
)
    
    return chain_with_cli_history

if __name__ == "__main__":
    dummy_input = {"messages": [{"role": "user", "content": "How do i unpack a tarball into another folder"}]}
    chain = build_chain()
    initial_response = chain.invoke(dummy_input)

    print(initial_response)