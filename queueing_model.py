#!/bin/env python3
from pobtl_model_checker import Model, Prop, EF, AG, StrongImplies, eval_formula
import numpy as np

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
    
    # Handle boundary cases
    if current_queue == 0:
        # At queue=0, can only stay or have arrival
        state_probabilities[current_state] = STAY + MU  # Absorb MU into STAY since queue empty
        next_state = frozenset({"Queue": current_queue + 1}.items())
        state_probabilities[next_state] = LAMBDA
    elif current_queue == 5:
        # At queue=5, can only stay or have departure
        # LAMBDA probability represents drops when queue is full
        state_probabilities[current_state] = STAY
        prev_state = frozenset({"Queue": current_queue - 1}.items())
        state_probabilities[prev_state] = MU
        drops = LAMBDA  # Track drop probability when queue full
    else:
        # Normal case: can stay, arrive, or depart
        state_probabilities[current_state] = STAY
        next_state = frozenset({"Queue": current_queue + 1}.items())
        state_probabilities[next_state] = LAMBDA
        prev_state = frozenset({"Queue": current_queue - 1}.items())
        state_probabilities[prev_state] = MU
    
    possible_transitions = list(state_probabilities.keys())
    transitions[current_state] = possible_transitions
    probabilities[current_state] = state_probabilities

# Build the model
model = Model(states, transitions)
model.probabilities = probabilities

# Create transition probability matrix P
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
drop_rate = drops * steady_state[-1]  # Probability of being in state 5 * lambda

print(f"\nAverage Queue Length: {avg_queue:.4f}")
print(f"Drop Probability: {drop_rate:.4f}")

# Propositions
q5 = Prop("Queue==5", lambda s: s["Queue"] == 5)
q4 = Prop("Queue==4", lambda s: s["Queue"] == 4)

# Strong implication: possibly queue reaches 4, and if so, then eventually reaches 5
formula = StrongImplies(q4, q5)

matching = eval_formula(formula, model)
print("\n✅ StrongImplies(Queue==4 → Queue==5):", matching)
