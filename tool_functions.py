#!/bin/env python3

# tool_functions.py
import os
from typing import Optional

def set_project(project_name: str) -> str:
    """
    Set the PROJECT environment variable.
    
    Args:
        project_name: The name of the project to set
        
    Returns:
        A confirmation message about the project change
    """
    old_project: Optional[str] = os.getenv("PROJECT")
    os.environ["PROJECT"] = project_name
    
    if old_project and old_project != project_name:
        return f"Switched project from {old_project} to {project_name}"
    return f"Set project to {project_name}"