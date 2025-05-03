
import os
import datetime
import sys
from openai import OpenAI
from tool_functions import set_project

project = os.getenv("PROJECT", "default")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    sys.stderr.write("❌ ERROR: OPENAI_API_KEY is not set in the environment.\n")
    sys.exit(1)

client = OpenAI()
requirements_file = lambda: f"{project}-REQUIREMENTS.md"

def append_to_requirements(role, message):
    timestamp = datetime.datetime.now().isoformat()
    with open(requirements_file(), "a") as f:
        f.write(f"\n## {role.capitalize()} @ {timestamp}\n\n")
        f.write(f"{message.strip()}\n")

def log_pobtl_translation(english, logic):
    with open(requirements_file(), "a") as f:
        f.write("\n<!-- POBTL* Translation -->\n")
        f.write(f"**English:** {english}\n\n")
        f.write(f"```pobtl\n{logic}\n```\n")

def load_readme_context():
    try:
        with open("README.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# POBTL* README not found."

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
}]

def chat():
    global project
    print(f"📂 Project: {project}")

    readme_context = load_readme_context()
    base_prompt = (
        """You are a modal logic and model construction assistant.

Before translating any requirements into POBTL* formulas, help the user build a discrete-event transition system model in Python. The system will be a Kripke-style state machine where each state is a combination of variable assignments, and each transition is a guarded update with a probability.

Once a model exists, then (and only then) translate user requirements into modal logic assertions using the POBTL* operators:
- EF, AG, AF, EG, EP, AH, etc.
- StrongImplies(p, q) = EF(p) and AG(p -> q)

All logic must be written in fenced Python code blocks using the label `pobtl`, and must be checkable using eval_formula() from the POBTL* library.

Your primary job is to help the user define, simulate, and analyze the logic of their system.

Here is the POBTL* language specification:

""" + readme_context +
        """

Begin modeling the user's system."""
    )

    print("🧠 Injecting README.md into the system prompt.")
    append_to_requirements("system", base_prompt)

    messages = [{"role": "system", "content": base_prompt}]
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower().strip() in {"quit", "exit"}:
                break
            append_to_requirements("user", user_input)
            messages.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == "set_project":
                        args = eval(tool_call.function.arguments)
                        result = set_project(args["project_name"])
                        project = os.getenv("PROJECT")
                        print(f"🔧 {result}")
                        append_to_requirements("system", result)
                        messages.append({
                            "role": "function",
                            "name": "set_project",
                            "content": result
                        })
                continue

            reply = msg.content.strip()
            print(f"🤖: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
            append_to_requirements("assistant", reply)

            if "must" in user_input or "should" in user_input:
                log_pobtl_translation(user_input, "AG(Implies(cond, effect))  # TODO")

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    chat()
