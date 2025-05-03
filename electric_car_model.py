#!/bin/env python3
from pobtl_model_checker import Model, Prop, EF, AF, AG, eval_formula

# Define states for the electric car
states = [
    {"location": "home", "charge": 100, "lights": False},
    {"location": "road", "charge": 90, "lights": True},
    {"location": "station", "charge": 10, "lights": True},
    {"location": "station", "charge": 100, "lights": True},
]

# Define transitions (simplified example)
transitions = {
    frozenset(s.items()): [
        frozenset(t.items())
        for t in states
        if s != t and abs(s["charge"] - t["charge"]) <= 90
    ]
    for s in states
}

# Construct model
model = Model(states, transitions)

# Define a property: charge is 100
charged = Prop("fully_charged", lambda s: s["charge"] == 100)

# Check if it's possible to eventually be fully charged
result = eval_formula(EF(charged), model)
print("✅ EF(charge == 100):", result)
