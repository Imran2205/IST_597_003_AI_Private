import os
import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("terminal")

user_home = os.getenv('HOME')
proc = None
# subprocess.Popen(
#     ["/bin/bash"],
#     cwd=user_home,
#     stdin=subprocess.PIPE,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     text=True,
#     bufsize=1
# )

async def run_in_terminal(cmd):
    global proc
    if proc is not None:
        if not cmd.endswith("\n"):
            cmd = cmd + "\n"

        marker = "[END_OF_CMD]"
        proc.stdin.write(cmd)
        proc.stdin.write(f"echo {marker}\n")
        proc.stdin.flush()

        output = []
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if marker in line:
                break
            output.append(line.rstrip())

        return output
    else:
        return ["No terminal has been initiated. Please initiate a terminal first with `initiate_terminal(working_dir)`"]

def format_output(input: dict) -> str:
    return f"""
Command: {input.get('input_command', 'Unknown')}
Output: {input.get('terminal_output', 'Unknown')}
"""

@mcp.tool()
async def initiate_terminal(cwd=None):
    """
    Initiate a new terminal.

    Args:
        cwd: directory where the terminal is initiated
    """
    global proc
    if proc is not None:
        return "Terminal is already open."

    if cwd is not None:
        proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    else:
        proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=user_home,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    return "Terminal Initiated"

@mcp.tool()
async def terminate_terminal():
    """
    Terminates the opened terminal
    """
    global proc
    if proc is not None:
        proc.terminate()
        proc.wait()
        return "Terminal Terminated"
    return "No terminal is open."

@mcp.tool()
async def run_command(command) -> str:
    """Runs a command in the terminal and return the output as a string

    Args:
        command: bash command to run in terminal
    """
    output = await run_in_terminal(command)
    out_dict = {
        "terminal_output": "\n".join(output),
        "input_command": command
    }
    final_out = format_output(out_dict)

    return final_out


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')