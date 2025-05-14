#!/bin/env python3
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from pobtl_model_checker import *
import sys
import graphviz
import random
import os

def markdown_to_python(md_content: str) -> str:
    """Convert markdown file to executable Python"""
    python_code = [
        "#!/usr/bin/env python3",
        "# This file is both markdown and executable Python",
        "# Generated from requirements parser",
        "",
        "import graphviz",
        "from dataclasses import dataclass",
        "from typing import Dict, List, Tuple, Any",
        "",
        "# Initialize documentation string",
        "DOC = []",
        "",
    ]

    # Split content into lines
    lines = md_content.split('\n')
    in_python_block = False
    in_temporal_block = False
    
    for line in lines:
        if line.startswith('```python'):
            in_python_block = True
            python_code.append("# Begin state machine definition")
            continue
        elif line.startswith('```temporal_logic'):
            in_temporal_block = True
            python_code.append("# Begin temporal logic")
            python_code.append("TEMPORAL_LOGIC = [")
            continue
        elif line.startswith('```'):
            if in_python_block:
                in_python_block = False
                python_code.append("# End state machine definition")
                python_code.append("")
                # Add code to generate diagram
                python_code.append("if 'model' in locals():")
                python_code.append("    model.generate_dot(project_name + '-graph')")
                python_code.append("")
            elif in_temporal_block:
                in_temporal_block = False
                python_code.append("]")
                python_code.append("")
            continue
        
        if in_python_block:
            python_code.append(line)
        elif in_temporal_block:
            if line.strip():
                python_code.append(f"    {repr(line)},")
        else:
            python_code.append(f"DOC.append({repr(line)})")
    
    # Add main block
    python_code.extend([
        "",
        "if __name__ == '__main__':",
        "    # Print documentation if --doc flag is provided",
        "    if '--doc' in sys.argv:",
        "        print('\\n'.join(DOC))",
        "    # Print temporal logic if --temporal flag is provided",
        "    elif '--temporal' in sys.argv:",
        "        print('Temporal Logic Statements:')",
        "        for stmt in TEMPORAL_LOGIC:",
        "        print(f'  {stmt}')",
        "    # Otherwise, show that the model was generated",
        "    else:",
        "        print(f'Generated state machine diagram: {project_name}-graph.png')",
    ])
    
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
        print(f"Reading file: {filename}")
        with open(filename, 'r') as f:
            content = f.read()
            
        # Extract project name
        project_match = re.search(r'Set project to (.*?)\n', content)
        if project_match:
            self.project_name = project_match.group(1).strip()
            print(f"Project name: {self.project_name}")
        
        # Find the first Python code block (state machine)
        state_machine_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
        state_machine_code = state_machine_match.group(1) if state_machine_match else EXAMPLE_STATE_MACHINE
        
        # Find all subsequent Python blocks (temporal logic)
        start_pos = state_machine_match.end() if state_machine_match else 0
        temporal_blocks = re.findall(r'```python\n(.*?)```', content[start_pos:], re.DOTALL)
        
        # Find all requirements
        requirements = []
        req_matches = re.findall(r'### REQ-(\w+)\s*\n(.*?)\n(?=###|\Z)', content, re.DOTALL)
        for req_id, desc in req_matches:
            requirements.append(Requirement(req_id, desc.strip(), None))
        
        return Conversation(
            project_name=self.project_name,
            state_machine_code=state_machine_code,
            temporal_logic_blocks=temporal_blocks,
            full_content=content,
            requirements=requirements
        )
    
    def execute_code(self, conversation: Conversation) -> None:
        """Execute the state machine and temporal logic code"""
        namespace = {
            'project_name': conversation.project_name,
            'Requirement': Requirement,
            'MarkovModel': MarkovModel
        }
        
        try:
            # Execute state machine code first
            print("\nExecuting state machine code...")
            exec(conversation.state_machine_code, namespace)
            
            # Generate diagram if model was created
            if 'model' in namespace:
                self.generate_diagram(namespace['model'])
            
            # Execute each temporal logic block
            print("\nExecuting temporal logic blocks...")
            for i, block in enumerate(conversation.temporal_logic_blocks, 1):
                print(f"\nTemporal Logic Block {i}:")
                exec(block, namespace)
                
        except Exception as e:
            print(f"Error executing code: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_diagram(self, model) -> None:
        """Generate and save the state machine diagram"""
        dot_file = f"{self.project_name}-graph.dot"
        png_file = f"{self.project_name}-graph.png"
        
        try:
            # Create a new directed graph
            dot = graphviz.Digraph(comment=f'State Machine for {self.project_name}')
            dot.attr(rankdir='TB')  # Top to bottom layout
            
            # Global graph attributes
            dot.attr('node', shape='box')  # Use boxes for states
            
            # Track initial state values
            initial_state = model.states[0] if model.states else {}
            initial_values = {k: v for k, v in initial_state.items()}
            
            # Add states as nodes
            for state in model.states:
                # Only show variables that differ from initial state
                diff_vars = []
                for k, v in state.items():
                    if k not in initial_values or initial_values[k] != v:
                        diff_vars.append(f"{k}={v}")
                
                # If no differences, just show "initial" for the first state
                label = '\n'.join(diff_vars) if diff_vars else "initial"
                if state == initial_state:
                    label = "initial\n" + '\n'.join(f"{k}={v}" for k, v in initial_values.items())
                
                state_id = str(hash(frozenset(state.items())))
                dot.node(state_id, label)
            
            # Add transitions as edges with probabilities and updates
            for state, next_states in model.transitions.items():
                state_id = str(hash(state))
                
                # Calculate probabilities (uniform if not specified)
                num_transitions = len(next_states)
                prob = 1.0 / num_transitions if num_transitions > 0 else 0
                
                for next_state in next_states:
                    next_id = str(hash(next_state))
                    
                    # Find variable updates
                    updates = []
                    state_dict = dict(state)
                    next_dict = dict(next_state)
                    
                    for k in state_dict:
                        if k in next_dict and state_dict[k] != next_dict[k]:
                            updates.append(f"{k}:={next_dict[k]}")
                    
                    # Create edge label with probability and updates
                    label = f"p={prob:.2f}"
                    if updates:
                        label += f"\n{', '.join(updates)}"
                    
                    dot.edge(state_id, next_id, label)
            
            # Save the dot file
            dot.save(dot_file)
            
            # Render PNG
            dot.render(filename=f"{self.project_name}-graph", format='png', cleanup=True)
            print(f"Generated {dot_file} and {png_file}")
            
            # Update image reference in markdown
            self.update_image_reference(png_file)
            
        except Exception as e:
            print(f"Error generating diagram: {e}")
            import traceback
            traceback.print_exc()
    
    def update_image_reference(self, png_file: str) -> None:
        """Update the markdown file with the current diagram"""
        if not self.project_name:
            return
            
        md_file = f"{self.project_name}-REQUIREMENTS.md"
        if not os.path.exists(md_file):
            return
            
        with open(md_file, 'r') as f:
            content = f.read()
            
        # Look for existing image reference
        image_pattern = r'!\[State Machine\]\(.*?\)'
        new_image_ref = f'![State Machine]({png_file})'
        
        if re.search(image_pattern, content):
            # Update existing image reference
            content = re.sub(image_pattern, new_image_ref, content)
        else:
            # Add new image reference after the first Python block
            content = re.sub(
                r'(```python.*?```)',
                f'\\1\n\n{new_image_ref}',
                content,
                flags=re.DOTALL,
                count=1  # Only replace first occurrence
            )
            
        with open(md_file, 'w') as f:
            f.write(content)

def verify_requirements(model: Any, requirements: List[Requirement]) -> Dict[str, bool]:
    """Verify all requirements against the model"""
    results = {}
    for req in requirements:
        if req.formula is not None:
            # Check formula in model
            satisfied = model.check(req.formula)
            results[req.id] = satisfied
    return results

class MarkovState:
    def __init__(self, name: str, precondition: str = "true"):
        self.name = name
        self.precondition = precondition
        self.transitions: Dict[str, Tuple[float, Dict[str, str]]] = {}  # target_state -> (probability, updates)

    def add_transition(self, target: str, probability: float = 1.0, updates: Dict[str, str] = None):
        """Add transition with probability and variable updates"""
        if updates is None:
            updates = {}
        self.transitions[target] = (probability, updates)

    def normalize_probabilities(self):
        """Ensure probabilities sum to 1.0"""
        total = sum(prob for prob, _ in self.transitions.values())
        if total == 0:
            return
        for target in self.transitions:
            prob, updates = self.transitions[target]
            self.transitions[target] = (prob / total, updates)

class MarkovModel:
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, MarkovState] = {}
        
    def add_state(self, name: str, precondition: str = "true") -> MarkovState:
        state = MarkovState(name, precondition)
        self.states[name] = state
        return state
    
    def generate_dot(self, filename: str):
        """Generate DOT file and PNG visualization"""
        dot = graphviz.Digraph(comment=f'State Machine for {self.name}')
        dot.attr(rankdir='LR')
        
        # Add states
        for state_name, state in self.states.items():
            label = f"{state_name}\n[{state.precondition}]"
            dot.node(state_name, label)
        
        # Add transitions
        for state_name, state in self.states.items():
            for target, (prob, updates) in state.transitions.items():
                label = f"{prob:.2f}"
                if updates:
                    label += "\n" + "\n".join(f"{k}:={v}" for k, v in updates.items())
                dot.edge(state_name, target, label)
        
        # Save both .dot and .png files
        dot_file = f"{filename}.dot"
        png_file = f"{filename}.png"
        dot.save(dot_file)
        dot.render(filename, format='png', cleanup=True)
        print(f"Generated {dot_file} and {png_file}")

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

    # Create Markov chain model
    model = MarkovModel(conversation.project_name)
    
    # Add states with preconditions
    initial = model.add_state("initial", "system.started == true")
    waiting = model.add_state("waiting_for_input", "input.available == false")
    processing = model.add_state("processing", "input.available == true")
    responding = model.add_state("responding", "processing.complete == true")
    error = model.add_state("error", "error.occurred == true")
    
    # Add transitions with probabilities and variable updates
    initial.add_transition("waiting_for_input", 1.0, {"system.ready": "true"})
    
    waiting.add_transition("processing", 0.7, {"input.processing": "true"})
    waiting.add_transition("waiting_for_input", 0.3)  # Stay in state
    
    processing.add_transition("responding", 0.8, {"processing.complete": "true"})
    processing.add_transition("error", 0.1, {"error.occurred": "true"})
    processing.add_transition("processing", 0.1)  # Stay in state
    
    responding.add_transition("waiting_for_input", 0.9, {
        "input.available": "false",
        "processing.complete": "false"
    })
    responding.add_transition("error", 0.1, {"error.occurred": "true"})
    
    error.add_transition("waiting_for_input", 0.8, {
        "error.occurred": "false",
        "system.ready": "true"
    })
    error.add_transition("error", 0.2)  # Stay in error state
    
    # Normalize all probabilities
    for state in model.states.values():
        state.normalize_probabilities()
    
    # Generate visualization
    model.generate_dot(f"{conversation.project_name}-graph")
    
    # Create Model for verification (convert MarkovModel to verification Model)
    states = []
    transitions = {}
    
    for state_name, state in model.states.items():
        state_dict = {"state": state_name}
        states.append(state_dict)
        current = frozenset(state_dict.items())
        transitions[current] = []
        
        for target, (prob, _) in state.transitions.items():
            if prob > 0:
                target_dict = {"state": target}
                transitions[current].append(frozenset(target_dict.items()))
    
    verification_model = Model(states, transitions)
    
    # Verify requirements
    print("\nVerifying requirements against model:")
    results = verify_requirements(verification_model, conversation.requirements)
    
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

model = MarkovModel(states, transitions) 