import subprocess

def get_command_history(command):
    history = apply_command(f"cat ~/.zsh_history | grep -m 5 '{command} '")
    
    history_array = []
    for line in history.stdout.readlines():
        history_array.append(line.decode("utf-8").split(";")[1].strip())
    
    return history_array

def apply_command(command):
    returned_value = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # returns the exit code in unix
    return returned_value