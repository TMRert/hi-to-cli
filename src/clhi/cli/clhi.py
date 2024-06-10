import click
from clhi.backend.chain import build_chain
import logging
import subprocess
import warnings

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.basicConfig(filename="logging.txt",level=logging.INFO)



@click.command(
    short_help="Entrypoint to the CLI AI assistent clHi!",
    epilog="This tool is built as part of the Databricks Data&AI Hackathon 2024.",
)
def hi():
    user_input = click.prompt("Ask me anything about the CLI!")

    question_buffer = [{"role": "user", "content": user_input}]
    formatted_input = {"messages": question_buffer}

    chain = build_chain()
    model_response = chain.invoke(formatted_input)
    click.echo(model_response)

    prompt_completed = False

    while not prompt_completed:
        click.echo(
            "Type [a] to apply the proposed CLI command, Type [f] to ask a follow-up question, type [q]' to quit the prompt"
        )
        c = click.getchar()
        if c.lower() == "a":
            output_lines = model_response.splitlines()
            for i in range(len(output_lines)):
                if "```" in output_lines[i]: # Handle cases where the model wraps the generated command in backticks
                    command = output_lines[i+1]

                    # Run command
                    if click.confirm(command):
                        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            else:
                click.echo("I did not find an executable command in my recommendation to apply.")
                break



        if c.lower() == "q":
            prompt_completed = True

        if c.lower() == "f":
            model_response
            question_buffer.append({"role": "assistant", "content": model_response})
            followup_input = click.prompt("How can I elaborate?")
            question_buffer.append({"role": "user", "content": followup_input})
            model_response = chain.invoke(formatted_input)
            click.echo(model_response)
