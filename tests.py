
from pobtl_model_checker import *

# A simple Kripke model for test cases
class Model:
    def __init__(self):
        self.states = {}
        self.transitions = {}
        self.predecessors = {}
        self.counter = 0

    def add_state(self, data):
        sid = self.counter
        self.states[sid] = data
        self.transitions[sid] = []
        self.predecessors[sid] = []
        self.counter += 1
        return sid

    def add_transition(self, src, dst, label=None):
        self.transitions[src].append((label, dst))
        self.predecessors[dst].append(src)

# Build a reasonable Kripke frame (nontrivial branching)
m = Model()
s0 = m.add_state({"p": False, "q": False})
s1 = m.add_state({"p": True, "q": False})
s2 = m.add_state({"p": True, "q": True})
s3 = m.add_state({"p": False, "q": True})
s4 = m.add_state({"p": True, "q": True})  # loop state
s5 = m.add_state({"p": False, "q": False})  # dead-end

m.add_transition(s0, s1)
m.add_transition(s1, s2)
m.add_transition(s2, s3)
m.add_transition(s3, s4)
m.add_transition(s4, s2)  # cycle
m.add_transition(s1, s5)

# Propositions
P = Prop("P", lambda s: s.get("p", False))
Q = Prop("Q", lambda s: s.get("q", False))

# Test cases
tests = {
    "EF(P)": EF(P),
    "AF(Q)": AF(Q),
    "AG(P)": AG(P),
    "EG(Q)": EG(Q),
    "EP(P)": EP(P),
    "AP(P)": AP(P),
    "EH(Q)": EH(Q),
    "AH(Q)": AH(Q),
    "StrongImplies(P → Q)": StrongImplies(P, Q),
    "¬EF(P)": Not(EF(P)),
    "P ∧ Q": And(P, Q),
    "P ∨ Q": Or(P, Q),
    "P → Q": Implies(P, Q),
}

def run_tests():
    for label, formula in tests.items():
        result = eval_formula(formula, m)
        result_states = sorted(list(result))
        print(f"{label}: holds in states {result_states}")

if __name__ == "__main__":
    run_tests()
