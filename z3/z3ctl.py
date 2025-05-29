from z3 import *
from typing import Dict, List, Set, Tuple, Callable
from enum import Enum
import itertools

class CTLOperator(Enum):
    """CTL temporal operators"""
    EX = "EX"  # Exists Next
    AX = "AX"  # All Next  
    EF = "EF"  # Exists Finally
    AF = "AF"  # All Finally
    EG = "EG"  # Exists Globally
    AG = "AG"  # All Globally
    EU = "EU"  # Exists Until
    AU = "AU"  # All Until

class CTLFormula:
    """Represents a CTL formula"""
    def __init__(self, operator=None, left=None, right=None, atomic=None):
        self.operator = operator
        self.left = left
        self.right = right
        self.atomic = atomic
    
    def __str__(self):
        if self.atomic:
            return str(self.atomic)
        elif self.operator in [CTLOperator.EX, CTLOperator.AX, CTLOperator.EF, 
                              CTLOperator.AF, CTLOperator.EG, CTLOperator.AG]:
            return f"{self.operator.value}({self.left})"
        elif self.operator in [CTLOperator.EU, CTLOperator.AU]:
            return f"{self.operator.value}({self.left}, {self.right})"
        else:
            return f"({self.left} {self.operator} {self.right})"

class StateMachine:
    """Represents a finite state machine"""
    def __init__(self):
        self.states: Set[str] = set()
        self.transitions: Dict[str, List[str]] = {}
        self.atomic_props: Dict[str, Set[str]] = {}  # prop -> set of states where it's true
        self.initial_states: Set[str] = set()
    
    def add_state(self, state: str):
        """Add a state to the machine"""
        self.states.add(state)
        if state not in self.transitions:
            self.transitions[state] = []
    
    def add_transition(self, from_state: str, to_state: str):
        """Add a transition between states"""
        self.add_state(from_state)
        self.add_state(to_state)
        if to_state not in self.transitions[from_state]:
            self.transitions[from_state].append(to_state)
    
    def add_atomic_prop(self, prop: str, states: List[str]):
        """Add an atomic proposition that holds in given states"""
        self.atomic_props[prop] = set(states)
        for state in states:
            self.add_state(state)
    
    def set_initial_states(self, states: List[str]):
        """Set the initial states"""
        self.initial_states = set(states)
        for state in states:
            self.add_state(state)

class CTLModelChecker:
    """CTL Model Checker using Z3"""
    
    def __init__(self, state_machine: StateMachine):
        self.sm = state_machine
        self.solver = Solver()
        self.state_vars = {}
        self.setup_z3_variables()
    
    def setup_z3_variables(self):
        """Create Z3 boolean variables for each state"""
        for state in self.sm.states:
            self.state_vars[state] = Bool(f"state_{state}")
    
    def evaluate_atomic(self, prop: str) -> Dict[str, bool]:
        """Evaluate atomic proposition in each state"""
        result = {}
        prop_states = self.sm.atomic_props.get(prop, set())
        for state in self.sm.states:
            result[state] = state in prop_states
        return result
    
    def evaluate_ex(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """EX φ: φ holds in some next state"""
        result = {}
        for state in self.sm.states:
            result[state] = any(phi_states[next_state] 
                              for next_state in self.sm.transitions.get(state, []))
        return result
    
    def evaluate_ax(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """AX φ: φ holds in all next states"""
        result = {}
        for state in self.sm.states:
            next_states = self.sm.transitions.get(state, [])
            if not next_states:  # No successors
                result[state] = True  # Vacuously true
            else:
                result[state] = all(phi_states[next_state] for next_state in next_states)
        return result
    
    def evaluate_ef(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """EF φ: φ holds eventually on some path"""
        # Fixed-point computation: EF φ = φ ∨ EX(EF φ)
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state]:
                    # Check if any successor satisfies EF φ
                    if any(result[next_state] 
                          for next_state in self.sm.transitions.get(state, [])):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_af(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """AF φ: φ holds eventually on all paths"""
        # Fixed-point computation: AF φ = φ ∨ AX(AF φ)
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state]:
                    next_states = self.sm.transitions.get(state, [])
                    if not next_states:
                        # No successors - φ will never hold
                        continue
                    
                    # Check if all successors satisfy AF φ
                    if all(result[next_state] for next_state in next_states):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_eg(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """EG φ: φ holds globally on some path"""
        # Fixed-point computation: EG φ = φ ∧ EX(EG φ)
        result = {state: True for state in self.sm.states}
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if result[state]:
                    # Must satisfy φ in current state
                    if not phi_states[state]:
                        new_result[state] = False
                        changed = True
                        continue
                    
                    # Must have some successor that satisfies EG φ
                    next_states = self.sm.transitions.get(state, [])
                    if not next_states or not any(result[next_state] for next_state in next_states):
                        new_result[state] = False
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_ag(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """AG φ: φ holds globally on all paths"""
        # Fixed-point computation: AG φ = φ ∧ AX(AG φ)
        result = {state: True for state in self.sm.states}
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if result[state]:
                    # Must satisfy φ in current state
                    if not phi_states[state]:
                        new_result[state] = False
                        changed = True
                        continue
                    
                    # All successors must satisfy AG φ
                    next_states = self.sm.transitions.get(state, [])
                    if next_states and not all(result[next_state] for next_state in next_states):
                        new_result[state] = False
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_eu(self, phi_states: Dict[str, bool], psi_states: Dict[str, bool]) -> Dict[str, bool]:
        """E[φ U ψ]: φ holds until ψ on some path"""
        # Fixed-point computation: E[φ U ψ] = ψ ∨ (φ ∧ EX(E[φ U ψ]))
        result = psi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state] and phi_states[state]:
                    # Check if any successor satisfies E[φ U ψ]
                    if any(result[next_state] 
                          for next_state in self.sm.transitions.get(state, [])):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_au(self, phi_states: Dict[str, bool], psi_states: Dict[str, bool]) -> Dict[str, bool]:
        """A[φ U ψ]: φ holds until ψ on all paths"""
        # Fixed-point computation: A[φ U ψ] = ψ ∨ (φ ∧ AX(A[φ U ψ]))
        result = psi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state] and phi_states[state]:
                    next_states = self.sm.transitions.get(state, [])
                    if not next_states:
                        # No successors - ψ will never hold
                        continue
                    
                    # All successors must satisfy A[φ U ψ]
                    if all(result[next_state] for next_state in next_states):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_formula(self, formula: CTLFormula) -> Dict[str, bool]:
        """Evaluate a CTL formula and return states where it holds"""
        if formula.atomic:
            return self.evaluate_atomic(formula.atomic)
        
        if formula.operator == CTLOperator.EX:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ex(phi_states)
        
        elif formula.operator == CTLOperator.AX:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ax(phi_states)
        
        elif formula.operator == CTLOperator.EF:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ef(phi_states)
        
        elif formula.operator == CTLOperator.AF:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_af(phi_states)
        
        elif formula.operator == CTLOperator.EG:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_eg(phi_states)
        
        elif formula.operator == CTLOperator.AG:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ag(phi_states)
        
        elif formula.operator == CTLOperator.EU:
            phi_states = self.evaluate_formula(formula.left)
            psi_states = self.evaluate_formula(formula.right)
            return self.evaluate_eu(phi_states, psi_states)
        
        elif formula.operator == CTLOperator.AU:
            phi_states = self.evaluate_formula(formula.left)
            psi_states = self.evaluate_formula(formula.right)
            return self.evaluate_au(phi_states, psi_states)
        
        else:
            raise ValueError(f"Unknown operator: {formula.operator}")
    
    def check_formula(self, formula: CTLFormula) -> bool:
        """Check if formula holds in all initial states"""
        result_states = self.evaluate_formula(formula)
        return all(result_states[state] for state in self.sm.initial_states)
    
    def get_satisfying_states(self, formula: CTLFormula) -> Set[str]:
        """Get all states where the formula is satisfied"""
        result_states = self.evaluate_formula(formula)
        return {state for state, satisfied in result_states.items() if satisfied}

# Helper functions for creating CTL formulas
def atomic(prop: str) -> CTLFormula:
    return CTLFormula(atomic=prop)

def EX(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EX, left=phi)

def AX(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.AX, left=phi)

def EF(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EF, left=phi)

def AF(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.AF, left=phi)

def EG(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EG, left=phi)

def AG(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.AG, left=phi)

def EU(phi: CTLFormula, psi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EU, left=phi, right=psi)

def AU(phi: CTLFormula, psi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.AU, left=phi, right=psi)

# Example usage
def example_traffic_light():
    """Example: Traffic light system"""
    # Create state machine
    sm = StateMachine()
    
    # Add states
    states = ["red", "yellow", "green"]
    for state in states:
        sm.add_state(state)
    
    # Add transitions (cyclic)
    sm.add_transition("red", "green")
    sm.add_transition("green", "yellow")
    sm.add_transition("yellow", "red")
    
    # Add atomic propositions
    sm.add_atomic_prop("stop", ["red"])
    sm.add_atomic_prop("go", ["green"])
    sm.add_atomic_prop("caution", ["yellow"])
    
    # Set initial state
    sm.set_initial_states(["red"])
    
    # Create model checker
    checker = CTLModelChecker(sm)
    
    # Check some CTL properties
    print("Traffic Light CTL Model Checking:")
    print("=" * 40)
    
    # AG(stop → AF go): If stopped, eventually can go
    stop_implies_af_go = CTLFormula()  # This would need proper boolean logic implementation
    
    # Simpler properties:
    # AF go: Eventually can go
    af_go = AF(atomic("go"))
    print(f"AF go (eventually can go): {checker.check_formula(af_go)}")
    print(f"States where AF go holds: {checker.get_satisfying_states(af_go)}")
    
    # AG EF stop: Always eventually stop
    ag_ef_stop = AG(EF(atomic("stop")))
    print(f"AG EF stop (always eventually stop): {checker.check_formula(ag_ef_stop)}")
    
    # EX go: Next state can be go
    ex_go = EX(atomic("go"))
    print(f"EX go (next can be go): {checker.check_formula(ex_go)}")
    print(f"States where EX go holds: {checker.get_satisfying_states(ex_go)}")

def example_mutual_exclusion():
    """Example: Mutual exclusion protocol"""
    sm = StateMachine()
    
    # States: (process1_state, process2_state)
    # States: idle, trying, critical
    states = [
        "idle_idle", "idle_trying", "idle_critical",
        "trying_idle", "trying_trying", "trying_critical",
        "critical_idle", "critical_trying"
        # Note: critical_critical is not allowed (mutual exclusion)
    ]
    
    for state in states:
        sm.add_state(state)
    
    # Add transitions (simplified Peterson's algorithm behavior)
    transitions = [
        ("idle_idle", "trying_idle"),
        ("idle_idle", "idle_trying"),
        ("trying_idle", "critical_idle"),
        ("trying_idle", "trying_trying"),
        ("idle_trying", "trying_trying"),
        ("idle_trying", "idle_critical"),
        ("trying_trying", "critical_trying"),
        ("trying_trying", "trying_critical"),
        ("critical_idle", "idle_idle"),
        ("idle_critical", "idle_idle"),
        ("critical_trying", "idle_trying"),
        ("trying_critical", "trying_idle")
    ]
    
    for from_state, to_state in transitions:
        sm.add_transition(from_state, to_state)
    
    # Atomic propositions
    sm.add_atomic_prop("p1_critical", ["critical_idle", "critical_trying"])
    sm.add_atomic_prop("p2_critical", ["idle_critical", "trying_critical"])
    sm.add_atomic_prop("p1_trying", ["trying_idle", "trying_trying", "trying_critical"])
    sm.add_atomic_prop("p2_trying", ["idle_trying", "trying_trying", "critical_trying"])
    
    sm.set_initial_states(["idle_idle"])
    
    checker = CTLModelChecker(sm)
    
    print("\nMutual Exclusion CTL Model Checking:")
    print("=" * 40)
    
    # Safety: AG ¬(p1_critical ∧ p2_critical) - never both in critical section
    # This would require boolean operators in CTL formulas
    
    # Liveness: AG(p1_trying → AF p1_critical) - if trying, eventually critical
    # Simplified check: AF p1_critical
    af_p1_critical = AF(atomic("p1_critical"))
    print(f"AF p1_critical (P1 eventually critical): {checker.check_formula(af_p1_critical)}")
    
    # Check reachability
    ef_p1_critical = EF(atomic("p1_critical"))
    print(f"EF p1_critical (P1 can reach critical): {checker.check_formula(ef_p1_critical)}")
    print(f"States where EF p1_critical holds: {checker.get_satisfying_states(ef_p1_critical)}")

if __name__ == "__main__":
    example_traffic_light()
    example_mutual_exclusion()
