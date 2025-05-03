# tool_functions.py
import os

def set_project(project_name: str) -> str:
    """
    Sets the PROJECT environment variable for logging.
    """
    os.environ["PROJECT"] = project_name
    return f"✅ Project switched to: {project_name}"