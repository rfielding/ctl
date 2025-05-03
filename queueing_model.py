#!/bin/env python3
from pobtl_model_checker import Model, Prop, EF, AG, StrongImplies, eval_formula

# Define queue states 0 to 5
states = [{"Queue": i} for i in range(6)]

# Allow increment or decrement by 1 within bounds
transitions = {
    frozenset(s.items()): [
        frozenset(t.items()) for t in states
        if abs(s["Queue"] - t["Queue"]) == 1 or s["Queue"] == t["Queue"]
    ]
    for s in states
}

# Build the model
model = Model(states, transitions)

# Propositions
q5 = Prop("Queue==5", lambda s: s["Queue"] == 5)
q4 = Prop("Queue==4", lambda s: s["Queue"] == 4)

# Strong implication: possibly queue reaches 4, and if so, then eventually reaches 5
formula = StrongImplies(q4, q5)

matching = eval_formula(formula, model)
print("✅ StrongImplies(Queue==4 → Queue==5):", matching)
