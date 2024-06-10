import click
from clhi.backend.chain import build_chain
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

@click.command(
    short_help="Entrypoint to the CLI AI assistent clHi!",
    epilog="This tool is built as part of the Databricks Data&AI Hackathon 2024.",
)
def hi():
    user_input = click.prompt("Ask my anything CLI!")
    formatted_input = {"messages": [{"role": "user", "content": user_input}]}
    click.echo(build_chain().invoke(formatted_input))

    # while not status["completed"]:
    #     ret = click.prompt(
    #             "Type [a] to apply the proposed CLI command, Type [f] to ask a follow-up question, type [q]' to quit the prompt")
    #         if lower(ret) == "a" or lower(ret) == "[a]":
                

    


