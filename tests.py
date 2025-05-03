#!/bin/env python3

from pobtl_model_checker import (
    Prop, EF, AF, AG, EG, EP, AP, EH, AH,
    And, Or, Not, Implies, Iff, StrongImplies,
    Model, eval_formula
)
import copy

def make_states():
    return [
        {"x": 0, "y": 10},
        {"x": 1, "y": 9},
        {"x": 2, "y": 8},
        {"x": 3, "y": 7},
        {"x": 4, "y": 6}
    ]

def make_transitions(states):
    transitions = {}
    for i in range(len(states) - 1):
        src = frozenset(copy.deepcopy(states[i]).items())
        dst = frozenset(copy.deepcopy(states[i + 1]).items())
        transitions[src] = [dst]
    transitions[frozenset(copy.deepcopy(states[-1]).items())] = []
    return transitions

def run_tests():
    states = make_states()  # plain dicts
    transitions = make_transitions(states)  # frozensets from copies
    model = Model(states, transitions)

    P = Prop("P", lambda s: s["x"] > 2)
    Q = Prop("Q", lambda s: s["y"] < 8)

    print("Testing modal formulas...")

    assert eval_formula(EF(P), model), "EF P failed"
    assert eval_formula(AF(Q), model), "AF Q failed"
    assert eval_formula(AG(Or(P, Not(P))), model), "AG(P ∨ ¬P) tautology failed"
    assert eval_formula(StrongImplies(P, Q), model), "StrongImplies(P, Q) failed"

    print("Testing temporal past operators (mock)...")
    reverse_transitions = {frozenset(copy.deepcopy(states[0]).items()): []}
    for i in range(1, len(states)):
        src = frozenset(copy.deepcopy(states[i]).items())
        prev = frozenset(copy.deepcopy(states[i - 1]).items())
        reverse_transitions[src] = [prev]
    past_model = Model(states, reverse_transitions)

    assert eval_formula(EP(P), past_model), "EP P failed"
    assert eval_formula(AP(Or(P, Not(P))), past_model), "AP(P ∨ ¬P) tautology failed"

    print("✅ All tests passed.")

if __name__ == "__main__":
    run_tests()