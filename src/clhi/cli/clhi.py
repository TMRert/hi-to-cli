from clhi.backend.chain import build_chain
from clhi.cli.utils import invoke_model, handle_user_response

import click
import questionary
from questionary import Choice

import logging
from textwrap import dedent


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.basicConfig(filename="logging.txt", level=logging.INFO)


from questionary import Style

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
    short_help="Entrypoint to the CLI AI assistent clHi!",
    epilog="This tool is built as part of the Databricks Data&AI Hackathon 2024.",
)
def hi():
    click.clear()
    chain = build_chain()
    context_buffer = []
    model_output = invoke_model(
        chain,
        context_buffer,
        cli_prompt="Hi, ask me anything about the CLI!",
        questionary_style=custom_style_fancy,
    )

    while True:
        user_resp = questionary.rawselect(
            "How would you like to proceed?",
            choices=[
                Choice("Appy the CLI Command", "a", shortcut_key="a"),
                Choice("Edit the CLI Command", "e", shortcut_key="e"),
                Choice("Ask follow-up question", "f", shortcut_key="f"),
                Choice("Quit", "q", shortcut_key="q"),
            ],
            style=custom_style_fancy
        ).ask()
        prompt_completed = handle_user_response(user_resp, model_output, context_buffer)
        if prompt_completed:
            break

        model_output = invoke_model(
            chain,
            context_buffer,
            cli_prompt="How can I help you further?",
            questionary_style=custom_style_fancy,
        )


if __name__ == "__main__":
    hi()