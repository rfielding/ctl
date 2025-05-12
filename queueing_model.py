#!/bin/env python3
from pobtl_model_checker import Model, Prop, EF, AG, StrongImplies, eval_formula
import numpy as np
from dataclasses import dataclass

# Define queue states 0 to 5
states = [{"Queue": i} for i in range(6)]

# Define transition probabilities
LAMBDA = 0.2  # arrival rate
MU = 0.2      # service rate
STAY = 0.6    # probability of staying in same state

# Allow increment or decrement by 1 within bounds
transitions = {}
probabilities = {}  # separate dictionary for probabilities
drops = 0  # Track theoretical drop probability

for s in states:
    current_queue = s["Queue"]
    current_state = frozenset(s.items())
    possible_transitions = []
    state_probabilities = {}
    
    # Always include staying in same state as a possibility
    possible_transitions.append(current_state)
    state_probabilities[current_state] = STAY
    
    # Add possible next states based on queue position
    if current_queue < 5:  # Can add to queue if not full
        next_state = frozenset({"Queue": current_queue + 1}.items())
        possible_transitions.append(next_state)
        state_probabilities[next_state] = LAMBDA
    
    if current_queue > 0:  # Can remove from queue if not empty
        prev_state = frozenset({"Queue": current_queue - 1}.items())
        possible_transitions.append(prev_state)
        state_probabilities[prev_state] = MU
    
    # For LTL checking, we need all possible transitions regardless of probability
    transitions[current_state] = possible_transitions
    # Keep probabilities separate for Markov chain analysis
    probabilities[current_state] = state_probabilities

# Build the model for LTL checking
model = Model(states, transitions)
model.probabilities = probabilities  # Store probabilities but don't use for LTL

# Create transition probability matrix P for Markov analysis
n_states = len(states)
P = np.zeros((n_states, n_states))

# Fill transition probability matrix
for i in range(n_states):
    current_state = frozenset({"Queue": i}.items())
    state_probs = probabilities[current_state]
    for next_state, prob in state_probs.items():
        j = dict(next_state)["Queue"]
        P[i][j] = prob

print("\nTransition Matrix P (rows=current state, cols=next state):")
print(P)
print("\nVerifying row sums (should be 1.0):")
print(np.sum(P, axis=1))

# Find steady state using power method
def get_steady_state(P, max_iter=1000, tol=1e-8):
    n = len(P)
    pi = np.ones(n) / n  # Initial uniform distribution
    
    for _ in range(max_iter):
        pi_next = pi @ P
        # Normalize to ensure probabilities sum to 1
        pi_next = pi_next / np.sum(pi_next)
        if np.allclose(pi, pi_next, rtol=tol):
            return pi_next
        pi = pi_next
    
    return pi

# Calculate steady state probabilities
steady_state = get_steady_state(P)

print("\nSteady State Probabilities (should sum to 1.0):")
for i, prob in enumerate(steady_state):
    print(f"P(Queue={i}) = {prob:.4f}")
print(f"Sum of probabilities: {sum(steady_state):.4f}")

# Calculate average queue length and drop probability
avg_queue = sum(i * prob for i, prob in enumerate(steady_state))
drop_rate = LAMBDA * steady_state[-1]  # Probability of being in state 5 * lambda

print(f"\nAverage Queue Length: {avg_queue:.4f}")
print(f"Drop Probability: {drop_rate:.4f}")

# CTL Properties and Checking
print("\nChecking CTL Properties:")

# Basic state properties
q5 = Prop("Queue==5", lambda s: s["Queue"] == 5)
q4 = Prop("Queue==4", lambda s: s["Queue"] == 4)
q0 = Prop("Queue==0", lambda s: s["Queue"] == 0)

# Check CTL properties (with path quantifiers E and A)
ctl_props = [
    ("EF(Queue==5)", EF(q5)),  # Exists path where eventually queue is full
    ("EF(Queue==0)", EF(q0)),  # Exists path where eventually queue is empty
    ("AG(Queue==4 → EF(Queue==5))", AG(StrongImplies(q4, q5))),  # All paths: from 4 can reach 5
]

print("\nCTL Properties:")
for desc, formula in ctl_props:
    matching = eval_formula(formula, model)
    print(f"✅ {desc}: {matching}")

# LTL properties would look like (if we had LTL support):
# F(Queue==5)      - Eventually queue is full
# G(Queue<5)       - Always queue is not full
# G(Queue==5 → F(Queue==4))  - Always: if full, eventually decreases
# F G(Queue<5)     - Eventually always not full

# Add LTL operators
@dataclass
class F:  # Finally (Eventually)
    f: any
    def eval(self, model, path):
        if isinstance(path, dict) or isinstance(path, frozenset):
            # Base case: evaluating a state property
            state = dict(path) if isinstance(path, frozenset) else path
            return self.f.eval(model, state)
        # Path case: check if property holds eventually in some suffix
        for i in range(len(path)):
            if self.eval(model, path[i]):  # Recursively evaluate on state
                return True
        return False

@dataclass
class G:  # Globally (Always)
    f: any
    def eval(self, model, path):
        if isinstance(path, dict) or isinstance(path, frozenset):
            # Base case: evaluating a state property
            state = dict(path) if isinstance(path, frozenset) else path
            return self.f.eval(model, state)
        # Path case: check if property holds for all states
        for i in range(len(path)):
            if not self.eval(model, path[i]):  # Recursively evaluate on state
                return False
        return True

@dataclass
class X:  # Next
    f: any
    def eval(self, model, path):
        if isinstance(path, dict) or isinstance(path, frozenset):
            return False  # Can't evaluate Next on a single state
        if len(path) < 2:
            return False
        return self.eval(model, path[1])  # Evaluate on next state

# Function to find paths in model
def get_paths(model, max_length=10):
    paths = []
    for state in model.states:
        path = [frozenset(state.items())]  # Start with frozenset
        paths.extend(extend_path(model, path, max_length))
    return paths

def extend_path(model, path, max_length):
    if len(path) >= max_length:
        return [path]
    paths = []
    current = path[-1]
    for next_state in model.transitions[current]:  # current is already frozenset
        new_path = path + [next_state]  # Keep as frozenset
        paths.extend(extend_path(model, new_path, max_length))
    return paths

# Function to check LTL formula on a path
def check_ltl_path(formula, model, path):
    return formula.eval(model, path)

# Example LTL properties
ltl_props = [
    ("F(Queue==5)", F(q5)),  # Eventually queue is full
    ("G(Queue<5)", G(Prop("Queue<5", lambda s: s["Queue"] < 5))),  # Always not full
    ("G(F(Queue==0))", G(F(q0))),  # Always eventually empty
]

print("\nLTL Properties:")
paths = get_paths(model)
for desc, formula in ltl_props:
    # Check formula on all paths
    satisfied = any(check_ltl_path(formula, model, path) for path in paths)
    print(f"✅ {desc}: {satisfied}")
