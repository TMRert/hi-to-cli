import logging
from textwrap import dedent

import click
import questionary
from questionary import Choice
from questionary import Style

from clhi.backend.chain import build_chain
from clhi.cli.utils import handle_user_response
from clhi.cli.utils import invoke_model

# Set logger to log to file to prevent debug output in the terminal.
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.basicConfig(filename="logging.txt", level=logging.INFO)

# define custom questionary style
custom_style_fancy = Style(
    [
        ("qmark", "fg:#673ab7 bold"),  # token in front of the question
        ("question", "bold"),  # question text
        ("answer", "fg:#f44336 bold"),  # submitted answer text behind the question
        ("pointer", "fg:#673ab7 bold"),  # pointer used in select and checkbox prompts
        ("highlighted", "fg:#673ab7 bold"),  # pointed-at choice in select and checkbox prompts
        ("selected", "fg:#cc5454"),  # style for a selected item of a checkbox
        ("separator", "fg:#cc5454"),  # separator in lists
        ("instruction", ""),  # user instructions for select, rawselect, checkbox
        ("text", ""),  # plain text
        ("disabled", "fg:#858585 italic"),  # disabled choices for select and checkbox prompts
    ]
)


@click.command(
    help=dedent(
        """
        AI assistant for the CLI that uses a RAG-type model architecture to help the user with generating commands and command explanations.
        Invokes the following Databricks model endpoints to generate responses:
        1. `databricks-gte-large-en` Embedding model to encode the user's query 
        2. Databricks Vector Search endpoint to find relevant documents related to the user's embedded query
        3. `databricks-dbrx-instruct` LLM to generate a response to the user's question.

        Enriches the LLM with relevant Linux Man documentation as well as the user's previous command history for related commands.   
        """
    ),
    short_help="Entrypoint to the CLI AI assistant; Say Hi to your CLI!",
    epilog="This tool is built as part of the Databricks Data&AI Hackathon 2024.",
)
def hi():

    # Clear view in the CLI
    click.clear()

    # Fetch langchain model and build empty context
    chain = build_chain()
    context_buffer = []

    # Prompt user for their first question.
    model_output = invoke_model(
        chain,
        context_buffer,
        cli_prompt="Hi, ask me anything about the CLI!",
        questionary_style=custom_style_fancy,
    )

    # While the user did not quit the tool, keep handling responses
    while True:

        # Based on the generated result, allow user to pick to apply, edit, continue prompting or quit
        user_resp = questionary.rawselect(
            "How would you like to proceed?",
            choices=[
                Choice("Appy the CLI Command", "a", shortcut_key="a"),
                Choice("Edit the CLI Command", "e", shortcut_key="e"),
                Choice("Ask follow-up question", "f", shortcut_key="f"),
                Choice("Quit", "q", shortcut_key="q"),
            ],
            style=custom_style_fancy,
        ).ask()
        prompt_completed = handle_user_response(user_resp, model_output, context_buffer)

        # if user exited the tool (either by command or exit code)
        if prompt_completed:
            break

        # if not, prompt for new question and generate new response.
        model_output = invoke_model(
            chain,
            context_buffer,
            cli_prompt="How can I help you further?",
            questionary_style=custom_style_fancy,
        )


# Define entrypoint into the CLI
if __name__ == "__main__":
    hi()
