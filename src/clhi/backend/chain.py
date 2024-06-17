from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from databricks.vector_search.client import VectorSearchClient
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables.base import RunnableSequence
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from operator import itemgetter

from typing import List, Tuple, Dict
from clhi.backend.terminal_utils import get_command_history

PROMPT_TEMPLATE = """You are an assistant for Linux Terminal users. You are answering questions related with terminal commands and your goal is to generate a relevant CLI command the user can run in their CLI.
Make sure to return the generated CLI command as the first line of your response, followed by an explanation of the command and possibly any arguments or option flags you used. Wrap or surround the generated command with a ```bash``` codeblock with backticks. Make sure the command is one-to-one copy-pastable into a terminal.

Question: {question}

You can use the following relevant Linux Manual documentation to provide a response: {summaries}
Use the user's historical CLI commands to generate customized recommendations: {commands}

Here is the chat history between you and the human: {chat_history}
"""


def get_vectorstore_retriever() -> VectorStoreRetriever:
    """Retrieve a langchain Vector Store Retriever model based on the Databricks embedding SDK. The vector store is powered by the `databricks-gte-large-en` large language model.

    Returns:
        VectorStoreRetriever: The searchable vector store object that allows for similarity search.
    """
    embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
    vsc = VectorSearchClient(disable_notice=True)

    vs_index = vsc.get_index(
        endpoint_name="dais_hackathon", index_name="workspace.default.man_7_info_index"
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="Text", embedding=embedding_model, columns=["Command", "Summary"]
    )

    return vectorstore.as_retriever()


def get_embeddings(user_input: str) -> Tuple[List[str], List[str]]:

    vectorstore = get_vectorstore_retriever()
    similar_documents = vectorstore.invoke(user_input)
    command_history = [x.metadata["Command"] for x in similar_documents]
    command_summary = [x.metadata["Summary"] for x in similar_documents]

    return command_history, command_summary


def extract_question(input: List[Dict[str, str]]) -> str:
    """Retrieve the latest asked user question to the AI assistent

    Args:
        input (List[Dict[str, str]]): Full input history and AI responses to provide to the model

    Returns:
        str: Latest user question without previous context
    """
    # The question is the last entry of the history
    return input[-1]["content"]


def extract_chat_history(input: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Retrieve the conversational history between the user and AI assistent without the user's latest question.

    Args:
        input (List[Dict[str, str]]): Full input history and AI responses to provide to the model

    Returns:
        List[Dict[str, str]]: Conversational history up ot the latest question prompted by the user. The dictionary contains the `role` key to indicate whether the prompt content came from the user or from the model.
    """
    return input[:-1]


def extract_cli_info(input: List[Dict[str, str]]) -> Dict[List[str], List[str]]:
    """Extract the historical CIL commands ran by the user and their manual descriptions related to the user's latest question. These commands are served as context to the model to help provide relevant generated CLI command responses tailored to the user.

    Args:
        input (List[Dict[str, str]]): Full input history and AI responses to provide to the model

    Returns:
        Dict[str, str]: Dictionary returning both the list of ran historical commands, as well as their Linux Man documentation.
    """

    # first extract latest question and get the commands similar to this question through our vector store
    question = extract_question(input)
    command_list, command_summaries = get_embeddings(question)
    output_array = []

    # for each command, fetch entries from the CLI history (if any) that were ran for this command
    for command in command_list:
        output_array.extend(get_command_history(command))
    return {"commands": output_array, "summaries": command_summaries}


def build_chain() -> RunnableSequence:
    """Build langchain chain for our RAG model. Chain consists out of our document retrieval using the `databricks-gte-large-en` embedding model and Vector Search index, Databricks DBRX instruct model, our custom prompt and a string output parser.

    Returns:
        RunnableSequence: The runnable Langchain chain that we can run the `invoke()` method on.
    """
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question", "commands", "summaries", "chat_history"],
    )
    chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=4096)

    chain_with_cli_history = (
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
            "cli_info": itemgetter("messages") | RunnableLambda(extract_cli_info),
        }
        | RunnableParallel(
            question=lambda x: x["question"],
            chat_history=lambda x: x["chat_history"],
            commands=lambda x: x["cli_info"]["commands"],
            summaries=lambda x: x["cli_info"]["summaries"],
        )
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain_with_cli_history


# Main method implemented for debugging purposes
if __name__ == "__main__":
    dummy_input = {
        "messages": [{"role": "user", "content": "How do i unpack a tarball into another folder"}]
    }
    chain = build_chain()
    initial_response = chain.invoke(dummy_input)

    print(initial_response)
