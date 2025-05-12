#!/bin/env python3
from pobtl_model_checker import Model, Prop, EF, AG, StrongImplies, eval_formula

# Define queue states 0 to 5
states = [{"Queue": i} for i in range(6)]

# Define transition probabilities
LAMBDA = 0.2  # arrival rate
MU = 0.2      # service rate
STAY = 0.6    # probability of staying in same state

# Allow increment or decrement by 1 within bounds
transitions = {}
probabilities = {}  # separate dictionary for probabilities

for s in states:
    current_queue = s["Queue"]
    current_state = frozenset(s.items())
    possible_transitions = []
    state_probabilities = {}
    
    # Stay in same state
    possible_transitions.append(current_state)
    state_probabilities[current_state] = STAY
    
    # Arrival (increment) if not at max capacity
    if current_queue < 5:
        next_state = frozenset({"Queue": current_queue + 1}.items())
        possible_transitions.append(next_state)
        state_probabilities[next_state] = LAMBDA
    
    # Departure (decrement) if queue not empty
    if current_queue > 0:
        prev_state = frozenset({"Queue": current_queue - 1}.items())
        possible_transitions.append(prev_state)
        state_probabilities[prev_state] = MU
    
    transitions[current_state] = possible_transitions
    probabilities[current_state] = state_probabilities

# Build the model
model = Model(states, transitions)
# Store probabilities as an attribute of the model for future use
model.probabilities = probabilities

# Propositions
q5 = Prop("Queue==5", lambda s: s["Queue"] == 5)
q4 = Prop("Queue==4", lambda s: s["Queue"] == 4)

# Strong implication: possibly queue reaches 4, and if so, then eventually reaches 5
formula = StrongImplies(q4, q5)

matching = eval_formula(formula, model)
print("✅ StrongImplies(Queue==4 → Queue==5):", matching)
