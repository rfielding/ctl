#!/bin/env python3
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from pobtl_model_checker import (
    Model, AG, EF, AF, EG,
    And, Or, Not, Implies
)
import sys

def markdown_to_python(md_content: str) -> str:
    """Convert markdown file to executable Python"""
    python_code = [
        "#!/usr/bin/env python3",
        "# Generated Python from markdown",
        "",
        "import graphviz",
        "from dataclasses import dataclass",
        "from typing import Dict, List, Tuple, Any",
        "from pobtl_model_checker import Model, MarkovModel, hashable",
        "",
    ]

    # Split content into lines
    lines = md_content.split('\n')
    in_python_block = False
    
    for line in lines:
        if line.startswith('```python'):
            in_python_block = True
            continue
        elif line.startswith('```'):
            if in_python_block:
                in_python_block = False
            continue
        
        if in_python_block:
            python_code.append(line)
    
    return '\n'.join(python_code)

@dataclass
class Requirement:
    id: str
    description: str
    formula: Any  # This will hold our temporal logic formulas

@dataclass
class Conversation:
    project_name: str
    state_machine_code: str
    temporal_logic_blocks: List[str]
    full_content: str
    requirements: List[Requirement]

class MarkovModel:
    """A model for a Markov Chain state machine"""
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions  # Dict[frozenset, List[Tuple[frozenset, float]]]
        self.normalize_probabilities()
    
    def normalize_probabilities(self):
        """Ensure probabilities sum to 1.0 for each state"""
        for state in self.transitions:
            total = sum(prob for _, prob in self.transitions[state])
            if total == 0:
                continue
            self.transitions[state] = [
                (next_state, prob/total) 
                for next_state, prob in self.transitions[state]
            ]
    
    def check(self, formula):
        """Check if a temporal logic formula holds in this model"""
        # Placeholder for temporal logic checking
        return True

class RequirementsParser:
    def __init__(self):
        self.project_name = ""
        self.requirements: List[Requirement] = []
        
    def parse_file(self, filename: str) -> Conversation:
        """Parse a markdown conversation file"""
        print(f"\n🔍 Reading file: {filename}")
        with open(filename, 'r') as f:
            content = f.read()
            
        print("\n=== Content Analysis ===")
        
        # Build up the complete Python code to execute
        python_code = [
            "# Generated Python from markdown",
            "from pobtl_model_checker import *  # Import all operators",
            "",
        ]
        
        # Extract project name
        project_match = re.search(r'Set project to (.*?)\n', content)
        if project_match:
            self.project_name = project_match.group(1).strip()
            print(f"📁 Project name: {self.project_name}")
            python_code.append(f"project_name = '{self.project_name}'")
        else:
            print("⚠️  No project name found")
        
        # Find Python code blocks that define the model
        python_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        if python_blocks:
            print(f"📊 Found {len(python_blocks)} Python code blocks")
            for block in python_blocks:
                python_code.append("\n# Model definition")
                python_code.append(block)
        else:
            print("⚠️  No Python code blocks found")
        
        # Find temporal logic blocks and convert them to assertions
        temporal_blocks = re.findall(r'```temporal\n(.*?)```', content, re.DOTALL)
        if temporal_blocks:
            print(f"🔬 Found {len(temporal_blocks)} temporal logic blocks")
            python_code.append("\n# Temporal logic assertions")
            for i, formula in enumerate(temporal_blocks, 1):
                # Clean up the formula text
                formula_lines = [line.strip() for line in formula.strip().split('\n') if line.strip()]
                
                # Extract comments
                comments = [line for line in formula_lines if line.startswith('#')]
                # Get code lines (non-comments)
                code_lines = [line for line in formula_lines if not line.startswith('#')]
                
                # Print description
                if comments:
                    python_code.append(f"\nprint('Formula {i} description:')")
                    for comment in comments:
                        python_code.append(f"print('{comment}')")
                
                # Add the actual formula evaluation
                python_code.append(f"\nprint('Evaluating formula {i}:')")
                
                # Handle each line of code separately
                for line in code_lines[:-1]:  # All lines except the last one
                    python_code.append(line)
                
                # The last line should be the actual formula
                formula_name = f"formula_{i}"
                python_code.append(f"{formula_name} = {code_lines[-1]}")
                
                # Evaluate the formula
                python_code.append(f"result_{i} = [{formula_name}.eval(model, state) for state in model.states]")
                python_code.append(f"print('Results for all states:', result_{i})")
                python_code.append(f"for i, (state, result) in enumerate(zip(model.states, result_{i})):")
                python_code.append("    print(f'State {i}: {state}')")
                python_code.append("    print(f'Result: {result}\\n')")
        
        # Combine all code
        final_code = '\n'.join(python_code)
        print("\n=== Generated Python Code ===")
        print(final_code)
        
        # Execute the combined code
        print("\n=== Execution Results ===")
        namespace = {}
        try:
            exec(final_code, namespace)
            self.model = namespace.get('model')
            if self.model:
                print("\n✅ Successfully executed model and formulas")
            else:
                print("\n⚠️  No model object found after execution")
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
        
        return Conversation(
            project_name=self.project_name,
            state_machine_code=final_code,  # Store the complete generated code
            temporal_logic_blocks=temporal_blocks,
            full_content=content,
            requirements=[]
        )
    
    def execute_code(self, conversation: Conversation) -> None:
        """Execute the state machine and temporal logic code"""
        namespace = {
            'project_name': conversation.project_name,
            'Requirement': Requirement,
            'MarkovModel': MarkovModel,
            'Model': Model,
            # Add temporal logic operators
            'AG': AG,
            'EF': EF,
            'AF': AF,
            'EG': EG,
            'And': And,
            'Or': Or,
            'Not': Not,
            'Implies': Implies,
        }
        
        try:
            print("\nExecuting state machine code...")
            exec(conversation.state_machine_code, namespace)
            
            print("\nExecuting temporal logic blocks...")
            for i, block in enumerate(conversation.temporal_logic_blocks, 1):
                print(f"\nTemporal Logic Block {i}:")
                exec(block, namespace)
                
        except Exception as e:
            print(f"Error executing code: {e}")
            import traceback
            traceback.print_exc()

def verify_requirements(model: Any, requirements: List[Requirement]) -> Dict[str, bool]:
    """Verify all requirements against the model"""
    results = {}
    for req in requirements:
        if req.formula is not None:
            # Check formula in model
            satisfied = model.check(req.formula)
            results[req.id] = satisfied
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./requirements_parser.py <requirements_file.md>")
        sys.exit(1)
        
    filename = sys.argv[1]
    
    parser = RequirementsParser()
    conversation = parser.parse_file(filename)
    
    if conversation:
        parser.execute_code(conversation)
    else:
        print("Failed to parse conversation")

    # Create Model for verification (convert MarkovModel to verification Model)
    states = [
        {'Actor': 'Google DeepMind', 'creation_start': False, 'AI_exists': False, 'in_progress': False},
        {'Actor': 'Google DeepMind', 'creation_start': True, 'AI_exists': False, 'in_progress': True},
        {'Actor': 'Google DeepMind', 'creation_start': True, 'AI_exists': True, 'in_progress': False}
    ]

    # Define the transitions
    transitions = {}
    for i in range(len(states)-1):
        state_items = hashable(states[i])
        next_state_items = hashable(states[i+1])
        transitions[state_items] = [(next_state_items, 1.0)]

    # Create the model
    model = Model(states=states, transitions=transitions)
    
    # Verify requirements
    print("\nVerifying requirements against model:")
    results = verify_requirements(model, conversation.requirements)
    
    # Print results
    for req_id, satisfied in results.items():
        req = next(r for r in conversation.requirements if r.id == req_id)
        print(f"\nREQ-{req_id}: {'✓' if satisfied else '✗'}")
        print(f"Description: {req.description}")
        if not satisfied:
            print("Failed to satisfy requirement")

# Define initial state and possible values
initial_state = {
    "presidentUs": "Trump",
    "presidentSyria": "Assad",
    "sanctions": True,
    "war": False,
    "oil_price": "high"
}

# Define possible transitions with probabilities
def get_transitions(state):
    transitions = []
    
    # Sanctions can be lifted or imposed
    if state["sanctions"]:
        new_state = state.copy()
        new_state["sanctions"] = False
        transitions.append((new_state, 0.3))  # 30% chance sanctions are lifted
    else:
        new_state = state.copy()
        new_state["sanctions"] = True
        transitions.append((new_state, 0.2))  # 20% chance sanctions are imposed
    
    # President changes
    new_state = state.copy()
    new_state["presidentUs"] = "Other" if state["presidentUs"] == "Trump" else "Trump"
    transitions.append((new_state, 0.1))  # 10% chance president changes
    
    # War status can change
    new_state = state.copy()
    new_state["war"] = not state["war"]
    if new_state["war"]:
        new_state["oil_price"] = "very_high"  # War affects oil prices
    transitions.append((new_state, 0.1))  # 10% chance war status changes
    
    # Can stay in current state
    transitions.append((state.copy(), 0.3))  # 30% chance nothing changes
    
    return transitions

# Generate all possible states and transitions
states = [initial_state]
transitions = {}

# Build state space
for state in states:
    current = frozenset(state.items())
    if current not in transitions:
        transitions[current] = []
        for next_state, prob in get_transitions(state):
            next_frozen = frozenset(next_state.items())
            transitions[current].append((next_frozen, prob))
            if next_state not in states:
                states.append(next_state)

model = MarkovModel(states=states, transitions=transitions) 
