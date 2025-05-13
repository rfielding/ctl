#!/bin/env python3
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from pobtl_model_checker import *
import sys

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
            # Example: "queue is empty" -> check Queue == 0
            if "empty" in key:
                self.props[key] = Prop(key, lambda s: s.get("Queue", 0) == 0)
            # Add more state patterns here
            
        return self.props.get(key, Prop(key, lambda s: True))

def verify_requirements(model: Model, requirements: List[Requirement]) -> Dict[str, bool]:
    """Verify all requirements against the model"""
    results = {}
    for req in requirements:
        # Check formula in initial states
        satisfied = all(req.formula.eval(model, state) for state in model.states)
        results[req.id] = satisfied
    return results

# Example usage:
if __name__ == "__main__":
    # Get filename from command line argument, default to "REQUIREMENTS.md" if not provided
    filename = sys.argv[1] if len(sys.argv) > 1 else "REQUIREMENTS.md"
    
    parser = RequirementsParser()
    reqs = parser.parse_file(filename)
    
    # Create model (example)
    states = [{"Queue": i} for i in range(4)]
    transitions = {}
    for i in range(4):
        current = frozenset({"Queue": i}.items())
        targets = [current]  # Can stay
        if i > 0:
            targets.append(frozenset({"Queue": i-1}.items()))
        if i < 3:
            targets.append(frozenset({"Queue": i+1}.items()))
        transitions[current] = targets
    
    model = Model(states, transitions)
    
    # Verify requirements
    results = verify_requirements(model, reqs)
    
    # Print results
    for req_id, satisfied in results.items():
        req = next(r for r in reqs if r.id == req_id)
        print(f"\nREQ-{req_id}: {'✓' if satisfied else '✗'}")
        print(f"Description: {req.description}")
        if not satisfied:
            print("Failed to satisfy requirement") 