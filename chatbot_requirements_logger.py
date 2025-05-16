#!/bin/env python3

import os
import datetime
import sys
from typing import Callable, Optional, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tool_functions import set_project
from pobtl_model_checker import Model, hashable
import re

project: str = os.getenv("PROJECT", "default")
openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    sys.stderr.write("❌ ERROR: OPENAI_API_KEY is not set in the environment.\n")
    sys.exit(1)

client: OpenAI = OpenAI()
requirements_file: Callable[[], str] = lambda: f"{project}-REQUIREMENTS.md"

def append_to_requirements(role: str, message: str) -> None:
    timestamp: str = datetime.datetime.now().isoformat()
    with open(requirements_file(), "a") as f:
        f.write(f"\n## {role.capitalize()} @ {timestamp}\n\n")
        f.write(f"{message.strip()}\n")

def create_requirements_file(project: str) -> None:
    """Initialize empty requirements file"""
    with open(requirements_file(), "w") as f:
        f.write(f"\n## System @ {datetime.now().isoformat()}\n\n")
        f.write(f"Set project to {project}\n\n")

def log_pobtl_translation(english: str, logic: str) -> None:
    """Log a temporal logic translation to the requirements file"""
    with open(requirements_file(), "a") as f:
        f.write("\n<!-- Temporal Logic Translation -->\n")
        f.write(f"**English:** {english}\n\n")
        f.write("```temporal\n")
        f.write(f"{logic}\n")
        f.write("```\n")

def log_model_and_code(model: Model, python_code: str) -> None:
    """Log visualization and complete model code"""
    with open(requirements_file(), "a") as f:
        # Add visualization
        f.write("## Model Visualization\n\n")
        f.write(f"![State Machine Model]({project}-model.png)\n\n")
        
        # Add DOT representation
        dot_string = model_to_dot(model)
        f.write("```dot\n")
        f.write(dot_string)
        f.write("\n```\n\n")
        
        # Always preserve complete model code
        f.write("## Model Definition\n\n")
        f.write("```python\n")
        f.write(python_code)
        f.write("\n```\n\n")

def load_readme_context() -> str:
    fullData: Optional[str] = None
    try:
        with open("pobtl_model_checker.py", "r") as f:
            fullData = f.read()
    except FileNotFoundError:
        return "# POBTL* README not found."
    try:
        with open("tests.py", "r") as f:
            fullData += f.read()
    except FileNotFoundError:
        return "# POBTL* README not found."
    return fullData if fullData else "# POBTL* README not found."

tools = [{
    "type": "function",
    "function": {
        "name": "set_project",
        "description": "Switch to a different project by setting the environment PROJECT variable",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Name of the project (used to generate REQUIREMENTS.md)"
                }
            },
            "required": ["project_name"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "graphviz_render",
        "description": "Render a graphviz DOT diagram of the model",
        "parameters": {
            "type": "object",
            "properties": {
                "dot_string": {
                    "type": "string",
                    "description": "The graphviz DOT format string to render"
                }
            },
            "required": ["dot_string"]
        }
    }
}]

def chat() -> None:
    global project
    print(f"📂 Project: {project}")

    readme_context: str = load_readme_context()
    base_prompt: str = (
        """You are a modal logic and model construction assistant. 
        
        Start by asking the user the name of the project we are working on.
        Once you know the project name, you can start to persist this conversation into
        a REQUIREMENTS.md file.

        This file exists so that the content can be reloaded into the chat history.
        Every prompt and response will have a header that specifies the role (user, assistant, system).

        When the system is asked to implement requirements, a Model will be created
        to model how the world evolves in time, through variable changes with probabilities.
        From this model, we will generate a png that represents the Markov Chain that
        temporal logic runs against.

        Whenever this model changes, a link to a freshly generated png from dot from the model will be added into this file as an inline image.
        Then, a single Python code fence that contains all of the model, plus temporal logic assertions will be added.
        Given this, ./requirements_parser.py should be able to execute this requirements file to check the results.
 
The temporal logic assertions using the POBTL* operators:
- EF, AG, AF, EG, EP, AH, etc.
- StrongImplies(p, q) = EF(p) and AG(p -> q)

YOU MUST RENDER THE PYTHON CODE FENCE OF THE MODEL WITH ASSERTIONS AND THE PNG;
Otherwise, the data will just get lost. That is because we are using this markdown file
as our requirements database.
This code is what you need to invoke in order to make the Model, and the assertions:

""" + readme_context +
        """

Begin modeling the user's system."""
    )

    print("🧠 Injecting README.md into the system prompt.")
    append_to_requirements("system", base_prompt)

    messages: list[dict[str, str]] = [{"role": "system", "content": base_prompt}]
    while True:
        try:
            user_input: str = input("\nYou: ")
            if user_input.lower().strip() in {"quit", "exit"}:
                break
            append_to_requirements("user", user_input)
            messages.append({"role": "user", "content": user_input})

            response: ChatCompletion = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            msg: ChatCompletionMessage = response.choices[0].message
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == "set_project":
                        args = eval(tool_call.function.arguments)
                        result = set_project(args["project_name"])
                        project = os.getenv("PROJECT", "default")
                        print(f"🔧 {result}")
                        append_to_requirements("system", result)
                        messages.append({
                            "role": "function",
                            "name": "set_project",
                            "content": result
                        })
                    elif tool_call.function.name == "graphviz_render":
                        args = eval(tool_call.function.arguments)
                        result = graphviz_render(args["dot_string"])
                        print(f"📊 {result}")
                        append_to_requirements("system", result)
                        messages.append({
                            "role": "function",
                            "name": "graphviz_render",
                            "content": result
                        })
                continue

            reply: str = msg.content.strip()
            print(f"🤖: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
            append_to_requirements("assistant", reply)

            if "must" in user_input or "should" in user_input:
                log_pobtl_translation(user_input, "AG(Implies(cond, effect))  # TODO")

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

def graphviz_render(dot_string: str) -> str:
    """Renders a graphviz DOT diagram to a file and returns the filename"""
    try:
        from graphviz import Source
        s = Source(dot_string)
        filename = f"{project}-model"
        # Generate PNG file
        s.render(filename, format='png', cleanup=True)
        
        # Add the image reference to the requirements file
        with open(requirements_file(), "a") as f:
            f.write(f"\n## Model Visualization\n\n")
            f.write(f"![State Machine Model]({filename}.png)\n\n")
            f.write("```dot\n")
            f.write(dot_string)
            f.write("\n```\n\n")
            
        return f"✅ Generated {filename}.png and added to requirements"
    except Exception as e:
        return f"❌ Failed to render graphviz: {str(e)}"

def model_to_dot(model: Model) -> str:
    """Generate DOT showing state variables and transition details"""
    dot_lines = ["digraph {"]
    dot_lines.append("    node [shape=record];")
    
    # Show all variables in each state
    for i, state in enumerate(model.states):
        vars_str = "|".join(f"{k}={v}" for k, v in sorted(state.items()))
        dot_lines.append(f'    state_{i} [label="{{{vars_str}}}"];')
    
    # Show probabilities and variable changes on transitions
    for i, state in enumerate(model.states):
        state_hash = hashable(state)
        if state_hash in model.transitions:
            for next_state_hash, prob in model.transitions[state_hash]:
                for j, next_state in enumerate(model.states):
                    if hashable(next_state) == next_state_hash:
                        changes = []
                        for var in state:
                            if state[var] != next_state[var]:
                                changes.append(f"{var}:={next_state[var]}")
                        change_str = ", ".join(changes) if changes else "no changes"
                        dot_lines.append(f'    state_{i} -> state_{j} [label="p={prob:.1f}\n{change_str}"];')
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

if __name__ == "__main__":
    chat()
