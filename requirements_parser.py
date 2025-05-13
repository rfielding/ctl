#!/bin/env python3
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from pobtl_model_checker import *
import sys
import graphviz
import random

@dataclass
class Requirement:
    id: str
    description: str
    formula: Formula

class RequirementsParser:
    def __init__(self):
        self.requirements: List[Requirement] = []
        self.props: Dict[str, Prop] = {}
        
    def parse_file(self, filename: str) -> List[Requirement]:
        """Parse requirements from a markdown file"""
        with open(filename, 'r') as f:
            content = f.read()
            
        # Find requirement blocks
        req_pattern = r'### REQ-(\w+)\s*\n(.*?)\n(?=###|\Z)'
        matches = re.finditer(req_pattern, content, re.DOTALL)
        
        for match in matches:
            req_id = match.group(1)
            description = match.group(2).strip()
            formula = self._description_to_formula(description)
            self.requirements.append(Requirement(req_id, description, formula))
            
        return self.requirements
    
    def _description_to_formula(self, desc: str) -> Formula:
        """Convert requirement description to temporal formula"""
        # Common patterns in requirements
        patterns = {
            r'always': AG,
            r'eventually': EF,
            r'after (.*?) then': lambda x: Implies(O(self._parse_state(x)), self._current_state()),
            r'must be followed by': lambda x, y: AG(Implies(x, EF(y))),
            r'never': lambda x: AG(Not(x)),
            r'before (.*?) occurs': lambda x, y: Implies(O(self._parse_state(x)), Not(y))
        }
        
        # Example conversion (to be expanded)
        if "always" in desc.lower():
            state_desc = re.sub(r'always', '', desc.lower()).strip()
            return AG(self._parse_state(state_desc))
        
        # Default to basic property
        return self._parse_state(desc)
    
    def _parse_state(self, desc: str) -> Formula:
        """Parse state description into formula"""
        # Create/reuse propositions based on description
        key = desc.lower().strip()
        if key not in self.props:
            # Handle different state descriptions
            if "waiting" in key:
                self.props[key] = Prop(key, lambda s: s.get("state") == "waiting_for_input")
            elif "processing" in key:
                self.props[key] = Prop(key, lambda s: s.get("state") == "processing")
            elif "responding" in key:
                self.props[key] = Prop(key, lambda s: s.get("state") == "responding")
            elif "error" in key:
                self.props[key] = Prop(key, lambda s: s.get("state") == "error")
            else:
                # Default proposition
                self.props[key] = Prop(key, lambda s: True)
            
        return self.props.get(key, Prop(key, lambda s: True))

def verify_requirements(model: Model, requirements: List[Requirement]) -> Dict[str, bool]:
    """Verify all requirements against the model"""
    results = {}
    for req in requirements:
        # Check formula in initial states
        satisfied = all(req.formula.eval(model, state) for state in model.states)
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
    filename = sys.argv[1] if len(sys.argv) > 1 else "REQUIREMENTS.md"
    project_name = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
    
    print(f"Parsing requirements from: {filename}")
    parser = RequirementsParser()
    reqs = parser.parse_file(filename)
    
    # Create Markov chain model
    model = MarkovModel(project_name)
    
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
    model.generate_dot(f"{project_name}-graph")
    
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
    results = verify_requirements(verification_model, reqs)
    
    # Print results
    for req_id, satisfied in results.items():
        req = next(r for r in reqs if r.id == req_id)
        print(f"\nREQ-{req_id}: {'✓' if satisfied else '✗'}")
        print(f"Description: {req.description}")
        if not satisfied:
            print("Failed to satisfy requirement") 