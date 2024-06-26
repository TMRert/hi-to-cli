from typing import Dict
from typing import List
from typing import Tuple

import click
import questionary
from langchain_core.runnables.base import RunnableSequence
from questionary import Style

from clhi.backend.terminal_utils import apply_command


def invoke_model(
    chain: RunnableSequence,
    context_buffer: List[Dict[str, str]],
    cli_prompt: str = "How can I elaborate",
    questionary_style: Style = None,
) -> str:
    """Prompt the user for a question and use that, as well as previous context buffers, to invoke our langchain chain.

    Args:
        chain (RunnableSequence): LangChain model used to generate the responses
        context_buffer (List[Dict[str, str]]): Previous conversational history between user and model (if available)
        cli_prompt (str, optional): Question prompted to the user in the CLI tool. Defaults to "How can I elaborate".
        questionary_style (Style, optional): Custom style how the prompt should be displayed to the user in the CLI. Defaults to None.

    Returns:
        str: Output generated by the chain and printed to the user.
    """
    # prompt the user to input their question / input
    user_question_resp = questionary.text(cli_prompt, style=questionary_style).ask()

    # if the user did not provide a valid response (e.g. executed a CTRL-C command)
    if not user_question_resp:
        exit()

    # Append user's input to the context buffer
    context_buffer.append({"role": "user", "content": user_question_resp})

    # invoke model and enrich context buffer
    model_response = chain.invoke({"messages": context_buffer})
    context_buffer.append({"role": "assistant", "content": model_response})

    # output model response to the user
    questionary.print(model_response, style="bold fg:ansibrightyellow")
    return model_response


def extract_terminal_command(model_response: str) -> Tuple[bool, str]:
    """Finds and extracts the generated CLI command from the response generated by the model, such that we can apply it for the user if desired.

    Args:
        model_response (str): Output of the model to the user

    Returns:
        Tuple[bool, str]: Tuple containing a boolean for if we found a command (True if found) and a following string containing the command (None if not found).
    """

    # chop up response into several lines and find the CLI command wrapped in ``` backticks
    output_lines = model_response.splitlines()
    for i in range(len(output_lines)):

        # if we find a backtick wrapped line, assume the command is present in the next line
        if "```" in output_lines[i]:
            return True, output_lines[i + 1]

    return False, None


def handle_user_response(
    user_resp: str, model_response: str, context_buffer: List[Dict[str, str]]
) -> bool:
    """Helper method that executes the commands based on the user's response to our AI's answer. Returns a boolean indicating whether the user wants to quit (True) or continue applying commands / questions (False).

    Args:
        user_resp (str): User's response captured by the `questionary` library
        model_response (str): Latest response generated by the AI assistant
        context_buffer (List[Dict[str, str]]): Full conversational history between the user and AI assistant

    Returns:
        bool: _description_
    """
    user_command = user_resp.lower()

    match user_command:
        case None:  # if user exited the CLI tool using signal interupts (SIGINT, EOF, etc.)
            return True
        case "q":  # if user exited the CLI tool using our built-in option
            return True
        case "a":  # User wants to apply the generated command

            # First see if we can extract the command
            cmd_found, cmd = extract_terminal_command(model_response)
            if cmd_found:

                # Prompt user with confirmation
                apply_cmd = questionary.confirm(
                    f"Do you want to apply the following command:\n{cmd}",
                    default=True,
                    auto_enter=True,
                ).ask()

                if apply_cmd:  # apply if confirmed, else return to question-asking
                    apply_command(cmd)
                return False
            else:
                # didn't find a command to apply, return to user input prompting to allow user to elaborate
                questionary.text(
                    "I did not find an executable command in my recommendation to apply."
                )
                return False

        case "e":  # User wants to edit generated command

            # Extract command and open editor
            _, cmd = extract_terminal_command(model_response)
            edited_response = click.edit(cmd)

            # prompt user with confirmation of edited command
            edit_cmd = questionary.confirm(
                f"Do you want to apply the following command:\n{edited_response}",
                default=True,
                auto_enter=True,
            ).ask()

            if edit_cmd:  # apply if confirmed, else return to question-asking
                apply_command(edited_response)
                context_buffer.append(
                    {
                        "role": "user",
                        "content": f"I've corrected your generated command '{cmd}' to '{edited_response}",
                    }
                )
            return False

        case "f":  # user wants to ask follow-up question without applying / editing the command
            return False
