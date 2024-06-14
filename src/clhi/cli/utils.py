from clhi.backend.terminal_utils import apply_command

import click
import questionary

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logging.basicConfig(filename="logging.txt", level=logging.INFO)


def invoke_model(chain, context_buffer, cli_prompt="How can I elaborate", questionary_style=None):

    # prompt the user to input their question / input
    user_question_resp = questionary.text(cli_prompt, style=questionary_style).ask()

    if not user_question_resp: 
        exit()
    context_buffer.append({"role": "user", "content": user_question_resp})
    
    # invoke model and enrich context buffer
    model_response = chain.invoke({"messages": context_buffer})
    context_buffer.append({"role": "assistant", "content": model_response})

    questionary.print(model_response, style="bold fg:ansibrightyellow")
    return model_response


def extract_terminal_command(model_response):

    # chop up response into several lines and find the CLI command wrapped in ``` backticks
    output_lines = model_response.splitlines()
    for i in range(len(output_lines)):

        # if we find a backtick wrapped line, assume the command is present in the next line
        if ("```" in output_lines[i]):  
            return True, output_lines[i+1]

    return False, None

def handle_user_response(user_resp: str, model_response: str, context_buffer: List[Dict[str, str]]) -> bool:
    user_command = user_resp.lower()

    match user_command:
        case None: 
            return True
        case "q":
            return True
        case "a":
            cmd_found, cmd = extract_terminal_command(model_response)
            if cmd_found:
                apply_cmd = questionary.confirm(
                        f"Do you want to apply the following command:\n{cmd}",
                        default=True,
                        auto_enter=True,
                    ).ask()
                
                if apply_cmd:
                    apply_command(cmd)
                return False
            else:
                questionary.text("I did not find an executable command in my recommendation to apply.")
                return False
            
        case "e":
            _, cmd = extract_terminal_command(model_response)
            edited_response = click.edit(cmd)
            edit_cmd = questionary.confirm(
                f"Do you want to apply the following command:\n{edited_response}",
                default=True,
                auto_enter=True
            ).ask()
            
            if edit_cmd:
                apply_command(edited_response)
                context_buffer.append({"role": "user", "content": f"I've corrected your generated command '{cmd}' to '{edited_response}"})
            return False

        case "f":
            context_buffer.append({"role": "assistant", "content": model_response})
            followup_input = questionary.text("How can I elaborate?")
            context_buffer.append({"role": "user", "content": followup_input})
            return False
