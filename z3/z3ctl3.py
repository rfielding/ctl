from z3 import *
from typing import Dict, List, Set, Tuple, Callable, Optional
from enum import Enum
import itertools
import numpy as np

class CTLOperator(Enum):
    """CTL and PCTL temporal operators"""
    EX = "EX"  # Exists Next
    AX = "AX"  # All Next  
    EF = "EF"  # Exists Finally
    AF = "AF"  # All Finally
    EG = "EG"  # Exists Globally
    AG = "AG"  # All Globally
    EU = "EU"  # Exists Until
    AU = "AU"  # All Until
    # Probabilistic operators
    PX = "PX"  # Probabilistic Next
    PF = "PF"  # Probabilistic Finally  
    PG = "PG"  # Probabilistic Globally
    PU = "PU"  # Probabilistic Until

class CTLFormula:
    """Represents a CTL/PCTL formula"""
    def __init__(self, operator=None, left=None, right=None, atomic=None, 
                 prob_bound=None, prob_operator=None):
        self.operator = operator
        self.left = left
        self.right = right
        self.atomic = atomic
        # For probabilistic operators: P>=0.7[F safe], P<0.1[G error]
        self.prob_bound = prob_bound  # The probability threshold (0.7, 0.1)
        self.prob_operator = prob_operator  # The comparison operator (">=", "<", "=")
    
    def __str__(self):
        if self.atomic:
            return str(self.atomic)
        elif self.operator in [CTLOperator.PX, CTLOperator.PF, CTLOperator.PG]:
            op_name = self.operator.value[1]  # Remove 'P' prefix
            return f"P{self.prob_operator}{self.prob_bound}[{op_name}({self.left})]"
        elif self.operator == CTLOperator.PU:
            return f"P{self.prob_operator}{self.prob_bound}[{self.left} U {self.right}]"
        elif self.operator in [CTLOperator.EX, CTLOperator.AX, CTLOperator.EF, 
                              CTLOperator.AF, CTLOperator.EG, CTLOperator.AG]:
            return f"{self.operator.value}({self.left})"
        elif self.operator in [CTLOperator.EU, CTLOperator.AU]:
            return f"{self.operator.value}({self.left}, {self.right})"
        else:
            return f"({self.left} {self.operator} {self.right})"

class ProbabilisticStateMachine:
    """Represents a probabilistic finite state machine (Markov Chain)"""
    def __init__(self):
        self.states: Set[str] = set()
        # Probabilistic transitions: state -> [(next_state, probability), ...]
        self.transitions: Dict[str, List[Tuple[str, float]]] = {}
        self.atomic_props: Dict[str, Set[str]] = {}
        self.initial_states: Dict[str, float] = {}  # state -> initial probability
    
    def add_state(self, state: str):
        """Add a state to the machine"""
        self.states.add(state)
        if state not in self.transitions:
            self.transitions[state] = []
    
    def add_transition(self, from_state: str, to_state: str, probability: float = 1.0):
        """Add a probabilistic transition between states"""
        self.add_state(from_state)
        self.add_state(to_state)
        
        # Check if transition already exists and update, or add new
        for i, (existing_to, existing_prob) in enumerate(self.transitions[from_state]):
            if existing_to == to_state:
                self.transitions[from_state][i] = (to_state, probability)
                return
        
        self.transitions[from_state].append((to_state, probability))
    
    def normalize_transitions(self, state: str):
        """Normalize transition probabilities from a state to sum to 1.0"""
        if state not in self.transitions or not self.transitions[state]:
            return
        
        total_prob = sum(prob for _, prob in self.transitions[state])
        if total_prob > 0:
            self.transitions[state] = [
                (to_state, prob / total_prob) 
                for to_state, prob in self.transitions[state]
            ]
    
    def add_atomic_prop(self, prop: str, states: List[str]):
        """Add an atomic proposition that holds in given states"""
        self.atomic_props[prop] = set(states)
        for state in states:
            self.add_state(state)
    
    def set_initial_distribution(self, distribution: Dict[str, float]):
        """Set the initial probability distribution over states"""
        total_prob = sum(distribution.values())
        self.initial_states = {
            state: prob / total_prob for state, prob in distribution.items()
        }
        for state in distribution.keys():
            self.add_state(state)
    
    def set_uniform_initial_states(self, states: List[str]):
        """Set uniform initial distribution over given states"""
        prob = 1.0 / len(states)
        self.set_initial_distribution({state: prob for state in states})
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix"""
        state_list = sorted(list(self.states))
        n = len(state_list)
        state_to_idx = {state: i for i, state in enumerate(state_list)}
        
        matrix = np.zeros((n, n))
        for from_state, transitions in self.transitions.items():
            from_idx = state_to_idx[from_state]
            for to_state, prob in transitions:
                to_idx = state_to_idx[to_state]
                matrix[from_idx][to_idx] = prob
        
        return matrix, state_list

class PCTLModelChecker:
    """Probabilistic CTL Model Checker using Z3"""
    
    def __init__(self, state_machine: ProbabilisticStateMachine):
        self.sm = state_machine
        self.solver = Solver()
        self.state_vars = {}
        self.transition_matrix, self.state_list = self.sm.get_transition_matrix()
        self.state_to_idx = {state: i for i, state in enumerate(self.state_list)}
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
    
    def compute_probabilistic_next(self, phi_states: Dict[str, bool]) -> Dict[str, float]:
        """Compute P[X φ] - probability that φ holds in next state"""
        result = {}
        for state in self.sm.states:
            prob = 0.0
            for next_state, transition_prob in self.sm.transitions.get(state, []):
                if phi_states.get(next_state, False):
                    prob += transition_prob
            result[state] = prob
        return result
    
    def compute_probabilistic_until(self, phi_states: Dict[str, bool], 
                                  psi_states: Dict[str, bool], 
                                  max_steps: int = 1000) -> Dict[str, float]:
        """
        Compute P[φ U ψ] using iterative method
        P[φ U ψ] = ψ + φ * P[X(φ U ψ)]
        """
        # Initialize: P[φ U ψ]^0 = ψ
        prob_until = {state: float(psi_states.get(state, False)) for state in self.sm.states}
        
        for step in range(max_steps):
            new_prob_until = {}
            converged = True
            
            for state in self.sm.states:
                if psi_states.get(state, False):
                    # If ψ holds, probability is 1
                    new_prob_until[state] = 1.0
                elif not phi_states.get(state, False):
                    # If φ doesn't hold and ψ doesn't hold, probability is 0
                    new_prob_until[state] = 0.0
                else:
                    # φ holds but ψ doesn't: P[φ U ψ] = P[X(φ U ψ)]
                    prob = 0.0
                    for next_state, transition_prob in self.sm.transitions.get(state, []):
                        prob += transition_prob * prob_until[next_state]
                    new_prob_until[state] = prob
                
                # Check convergence
                if abs(new_prob_until[state] - prob_until[state]) > 1e-6:
                    converged = False
            
            prob_until = new_prob_until
            if converged:
                break
        
        return prob_until
    
    def compute_probabilistic_finally(self, phi_states: Dict[str, bool]) -> Dict[str, float]:
        """Compute P[F φ] = P[True U φ]"""
        true_states = {state: True for state in self.sm.states}
        return self.compute_probabilistic_until(true_states, phi_states)
    
    def compute_probabilistic_globally(self, phi_states: Dict[str, bool], 
                                     max_steps: int = 1000) -> Dict[str, float]:
        """
        Compute P[G φ] using the complement: P[G φ] = 1 - P[F ¬φ]
        """
        not_phi_states = {state: not phi_states.get(state, False) for state in self.sm.states}
        prob_finally_not_phi = self.compute_probabilistic_finally(not_phi_states)
        return {state: 1.0 - prob_finally_not_phi[state] for state in self.sm.states}
    
    # Keep the original CTL methods for non-probabilistic operators
    def evaluate_ex(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """EX φ: φ holds in some next state"""
        result = {}
        for state in self.sm.states:
            result[state] = any(phi_states.get(next_state, False)
                              for next_state, _ in self.sm.transitions.get(state, []))
        return result
    
    def evaluate_ax(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """AX φ: φ holds in all next states"""
        result = {}
        for state in self.sm.states:
            next_states = [ns for ns, _ in self.sm.transitions.get(state, [])]
            if not next_states:
                result[state] = True  # Vacuously true
            else:
                result[state] = all(phi_states.get(next_state, False) for next_state in next_states)
        return result
    
    def evaluate_ef(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """EF φ: φ holds eventually on some path"""
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state]:
                    if any(result.get(next_state, False)
                          for next_state, _ in self.sm.transitions.get(state, [])):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_af(self, phi_states: Dict[str, bool]) -> Dict[str, bool]:
        """AF φ: φ holds eventually on all paths"""
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.sm.states:
                if not result[state]:
                    next_states = [ns for ns, _ in self.sm.transitions.get(state, [])]
                    if not next_states:
                        continue
                    
                    if all(result.get(next_state, False) for next_state in next_states):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_formula(self, formula: CTLFormula) -> Dict[str, any]:
        """Evaluate a CTL/PCTL formula"""
        if formula.atomic:
            return self.evaluate_atomic(formula.atomic)
        
        # Standard CTL operators
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
        
        # Probabilistic operators
        elif formula.operator == CTLOperator.PX:
            phi_states = self.evaluate_formula(formula.left)
            probs = self.compute_probabilistic_next(phi_states)
            return self.apply_probability_bound(probs, formula.prob_operator, formula.prob_bound)
        
        elif formula.operator == CTLOperator.PF:
            phi_states = self.evaluate_formula(formula.left)
            probs = self.compute_probabilistic_finally(phi_states)
            return self.apply_probability_bound(probs, formula.prob_operator, formula.prob_bound)
        
        elif formula.operator == CTLOperator.PG:
            phi_states = self.evaluate_formula(formula.left)
            probs = self.compute_probabilistic_globally(phi_states)
            return self.apply_probability_bound(probs, formula.prob_operator, formula.prob_bound)
        
        elif formula.operator == CTLOperator.PU:
            phi_states = self.evaluate_formula(formula.left)
            psi_states = self.evaluate_formula(formula.right)
            probs = self.compute_probabilistic_until(phi_states, psi_states)
            return self.apply_probability_bound(probs, formula.prob_operator, formula.prob_bound)
        
        else:
            raise ValueError(f"Unknown operator: {formula.operator}")
    
    def apply_probability_bound(self, probabilities: Dict[str, float], 
                              operator: str, bound: float) -> Dict[str, bool]:
        """Apply probability bound to get boolean satisfaction"""
        result = {}
        for state, prob in probabilities.items():
            if operator == ">=":
                result[state] = prob >= bound
            elif operator == ">":
                result[state] = prob > bound
            elif operator == "<=":
                result[state] = prob <= bound
            elif operator == "<":
                result[state] = prob < bound
            elif operator == "=":
                result[state] = abs(prob - bound) < 1e-6
            else:
                raise ValueError(f"Unknown probability operator: {operator}")
        return result
    
    def get_probabilities(self, formula: CTLFormula) -> Dict[str, float]:
        """Get actual probabilities (not boolean satisfaction) for probabilistic formulas"""
        if formula.operator == CTLOperator.PX:
            phi_states = self.evaluate_formula(formula.left)
            return self.compute_probabilistic_next(phi_states)
        elif formula.operator == CTLOperator.PF:
            phi_states = self.evaluate_formula(formula.left)
            return self.compute_probabilistic_finally(phi_states)
        elif formula.operator == CTLOperator.PG:
            phi_states = self.evaluate_formula(formula.left)
            return self.compute_probabilistic_globally(phi_states)
        elif formula.operator == CTLOperator.PU:
            phi_states = self.evaluate_formula(formula.left)
            psi_states = self.evaluate_formula(formula.right)
            return self.compute_probabilistic_until(phi_states, psi_states)
        else:
            raise ValueError("Formula is not probabilistic")
    
    def check_formula(self, formula: CTLFormula) -> bool:
        """Check if formula holds in initial states"""
        result_states = self.evaluate_formula(formula)
        # Weighted check based on initial distribution
        total_prob = 0.0
        satisfying_prob = 0.0
        
        for state, init_prob in self.sm.initial_states.items():
            total_prob += init_prob
            if result_states.get(state, False):
                satisfying_prob += init_prob
        
        return satisfying_prob / total_prob > 0.99 if total_prob > 0 else False

# Helper functions for creating CTL/PCTL formulas
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

# Probabilistic CTL operators
def PX(phi: CTLFormula, operator: str, bound: float) -> CTLFormula:
    """P operator bound [X phi] - Probabilistic Next"""
    return CTLFormula(operator=CTLOperator.PX, left=phi, 
                     prob_operator=operator, prob_bound=bound)

def PF(phi: CTLFormula, operator: str, bound: float) -> CTLFormula:
    """P operator bound [F phi] - Probabilistic Finally"""
    return CTLFormula(operator=CTLOperator.PF, left=phi,
                     prob_operator=operator, prob_bound=bound)

def PG(phi: CTLFormula, operator: str, bound: float) -> CTLFormula:
    """P operator bound [G phi] - Probabilistic Globally"""
    return CTLFormula(operator=CTLOperator.PG, left=phi,
                     prob_operator=operator, prob_bound=bound)

def PU(phi: CTLFormula, psi: CTLFormula, operator: str, bound: float) -> CTLFormula:
    """P operator bound [phi U psi] - Probabilistic Until"""
    return CTLFormula(operator=CTLOperator.PU, left=phi, right=psi,
                     prob_operator=operator, prob_bound=bound)

# Example: Probabilistic Network Protocol
def example_probabilistic_network():
    """Example: Network with probabilistic failures and recovery"""
    print("Probabilistic Network Protocol PCTL Model Checking:")
    print("=" * 55)
    
    sm = ProbabilisticStateMachine()
    
    # States: (connection_status, data_status)
    states = ["connected_clean", "connected_corrupted", "disconnected_clean", 
              "disconnected_corrupted", "recovering"]
    
    for state in states:
        sm.add_state(state)
    
    # Probabilistic transitions
    # From connected_clean
    sm.add_transition("connected_clean", "connected_clean", 0.85)      # Stay stable
    sm.add_transition("connected_clean", "connected_corrupted", 0.1)   # Data corruption
    sm.add_transition("connected_clean", "disconnected_clean", 0.05)   # Network failure
    
    # From connected_corrupted  
    sm.add_transition("connected_corrupted", "connected_clean", 0.3)   # Auto-recovery
    sm.add_transition("connected_corrupted", "connected_corrupted", 0.4) # Stay corrupted
    sm.add_transition("connected_corrupted", "disconnected_corrupted", 0.3) # Disconnect
    
    # From disconnected states
    sm.add_transition("disconnected_clean", "recovering", 0.6)         # Try to reconnect
    sm.add_transition("disconnected_clean", "disconnected_clean", 0.4) # Stay disconnected
    
    sm.add_transition("disconnected_corrupted", "recovering", 0.8)     # Higher urgency
    sm.add_transition("disconnected_corrupted", "disconnected_corrupted", 0.2)
    
    # From recovering
    sm.add_transition("recovering", "connected_clean", 0.7)            # Successful recovery
    sm.add_transition("recovering", "connected_corrupted", 0.1)        # Partial recovery
    sm.add_transition("recovering", "disconnected_clean", 0.2)         # Recovery failed
    
    # Atomic propositions
    sm.add_atomic_prop("connected", ["connected_clean", "connected_corrupted", "recovering"])
    sm.add_atomic_prop("clean_data", ["connected_clean", "disconnected_clean"])
    sm.add_atomic_prop("operational", ["connected_clean"])
    sm.add_atomic_prop("critical_failure", ["disconnected_corrupted"])
    
    # Initial distribution - start mostly operational
    sm.set_initial_distribution({
        "connected_clean": 0.8,
        "connected_corrupted": 0.1,
        "disconnected_clean": 0.1
    })
    
    checker = PCTLModelChecker(sm)
    
    print("\nProbabilistic Properties:")
    print("-" * 30)
    
    # P>=0.9[F operational] - High probability of eventually being operational
    pf_operational_high = PF(atomic("operational"), ">=", 0.9)
    print(f"P>=0.9[F operational] (90%+ chance to become operational): {checker.check_formula(pf_operational_high)}")
    
    # Get actual probabilities
    pf_operational_probs = checker.get_probabilities(PF(atomic("operational"), ">=", 0.0))
    print("Actual P[F operational] probabilities by state:")
    for state in sorted(pf_operational_probs.keys()):
        print(f"  {state}: {pf_operational_probs[state]:.3f}")
    
    # P<0.1[G critical_failure] - Low probability of permanent critical failure
    pg_critical_low = PG(atomic("critical_failure"), "<", 0.1)
    print(f"\nP<0.1[G critical_failure] (< 10% chance of permanent critical failure): {checker.check_formula(pg_critical_low)}")
    
    # Get actual probabilities for staying in critical failure
    pg_critical_probs = checker.get_probabilities(PG(atomic("critical_failure"), ">=", 0.0))
    print("Actual P[G critical_failure] probabilities by state:")
    for state in sorted(pg_critical_probs.keys()):
        print(f"  {state}: {pg_critical_probs[state]:.3f}")
    
    # P>=0.8[clean_data U operational] - High probability that clean data leads to operational
    pu_clean_to_operational = PU(atomic("clean_data"), atomic("operational"), ">=", 0.8)
    print(f"\nP>=0.8[clean_data U operational] (80%+ chance clean data leads to operational): {checker.check_formula(pu_clean_to_operational)}")
    
    # P>=0.5[X connected] - At least 50% chance of being connected next
    px_connected = PX(atomic("connected"), ">=", 0.5)
    print(f"P>=0.5[X connected] (50%+ chance connected next step): {checker.check_formula(px_connected)}")
    
    # Get next-step connection probabilities
    px_connected_probs = checker.get_probabilities(PX(atomic("connected"), ">=", 0.0))
    print("Actual P[X connected] probabilities by state:")
    for state in sorted(px_connected_probs.keys()):
        print(f"  {state}: {px_connected_probs[state]:.3f}")

def example_probabilistic_vs_deterministic():
    """Compare probabilistic and deterministic properties"""
    print("\n\nProbabilistic vs Deterministic Comparison:")
    print("=" * 45)
    
    sm = ProbabilisticStateMachine()
    
    # Simple 3-state system with probabilistic recovery
    states = ["good", "degraded", "failed"]
    for state in states:
        sm.add_state(state)
    
    # Transitions with probabilities
    sm.add_transition("good", "good", 0.9)
    sm.add_transition("good", "degraded", 0.1)
    
    sm.add_transition("degraded", "good", 0.4)      # Recovery chance
    sm.add_transition("degraded", "degraded", 0.3)
    sm.add_transition("degraded", "failed", 0.3)
    
    sm.add_transition("failed", "degraded", 0.2)    # Repair chance  
    sm.add_transition("failed", "failed", 0.8)
    
    sm.add_atomic_prop("healthy", ["good"])
    sm.add_atomic_prop("broken", ["failed"])
    
    sm.set_uniform_initial_states(["good"])
    
    checker = PCTLModelChecker(sm)
    
    print("\nDeterministic CTL Properties:")
    print("-" * 25)
    ef_healthy = EF(atomic("healthy"))
    print(f"EF healthy (can possibly reach healthy): {checker.check_formula(ef_healthy)}")
    
    af_healthy = AF(atomic("healthy"))
    print(f"AF healthy (will definitely reach healthy): {checker.check_formula(af_healthy)}")
    
    print("\nProbabilistic PCTL Properties:")  
    print("-" * 25)
    
    # Different probability thresholds for same property
    for threshold in [0.5, 0.7, 0.9, 0.99]:
        pf_healthy = PF(atomic("healthy"), ">=", threshold)
        result = checker.check_formula(pf_healthy)
        print(f"P>={threshold}[F healthy] (≥{threshold*100}% chance to reach healthy): {result}")
    
    # Show actual probabilities
    probs = checker.get_probabilities(PF(atomic("healthy"), ">=", 0.0))
    print(f"\nActual P[F healthy] probabilities:")
    for state in sorted(probs.keys()):
        print(f"  From {state}: {probs[state]:.3f}")

if __name__ == "__main__":
    example_probabilistic_network()
    example_probabilistic_vs_deterministic()
