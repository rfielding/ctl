# extract_and_run_pobtl.py
import re
import sys

def extract_and_run(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Match blocks like ```pobtl ... ```
    pobtl_blocks = re.findall(r"```pobtl\s*(.*?)```", content, re.DOTALL)
    print(f"🧠 Found {len(pobtl_blocks)} POBTL* code block(s). Executing...")

    context = {}
    for i, code in enumerate(pobtl_blocks, 1):
        print(f"\n--- Executing block {i} ---")
        try:
            exec(code, context)
        except Exception as e:
            print(f"❌ Error in block {i}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_and_run_pobtl.py <REQUIREMENTS.md>")
    else:
        extract_and_run(sys.argv[1])

