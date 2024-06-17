import subprocess
from typing import List


def get_command_history(command: str, max_return_size: int = 5) -> List[str]:
    """Fetch a configurable amount of CLI commands ran by the user from the terminal history. This method for now assumes the user uses ZShell as their terminal.

    Args:
        command (str): Command for which we should find past commands
        max_return_size (int, optional): Amount of historical commands to return. Defaults to 5.

    Returns:
        List[str]: List of historical commands executed by the user.
    """

    # Regex search in the zshell history file for the command, and return at most `max_return_size` commands
    history = apply_command(f"cat ~/.zsh_history | grep -m {max_return_size} '{command} '")

    history_array = []

    # Read the result and parse the commands. Command history is formatted with the time, so we need to strip the output
    # example of `line` contents: `: 1646217610:0;echo hi`

    for line in history.stdout.readlines():
        history_array.append(
            line.decode("utf-8")
            .split(";")[1]
            .strip()  # Decode the output, find the command after the semi-column and clean of trailing spaces
        )

    return history_array


def apply_command(command: str) -> subprocess.Popen:
    """Run a user-provided command in the terminal by using a sub-process invocation.

    Args:
        command (str): command to execute in the terminal.

    Returns:
        subprocess.Popen: Instance of the Popen class that holds the subprocesses' return code, stdout and stderr outputs.
    """

    # shell=True should be set since we don't provide the command as list
    returned_value = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )  # returns the exit code in unix
    return returned_value
