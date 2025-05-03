
# electric_car_model.py
from pobtl_model_checker import *

class Model:
    def __init__(self):
        self.states = {}        # id -> dict
        self.transitions = {}   # id -> list of (label, id)
        self.predecessors = {}  # id -> list of predecessor ids
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

m = Model()

# States: driving between chargers
s0 = m.add_state({"loc": "A", "charge": 3})
s1 = m.add_state({"loc": "road", "charge": 2})
s2 = m.add_state({"loc": "road", "charge": 1})
s3 = m.add_state({"loc": "B", "charge": 0})
s4 = m.add_state({"loc": "B", "charge": 3})  # recharged
s5 = m.add_state({"loc": "road", "charge": 2})

# Transitions
m.add_transition(s0, s1)
m.add_transition(s1, s2)
m.add_transition(s2, s3)
m.add_transition(s3, s4)  # recharging
m.add_transition(s4, s5)

# Formulas
low = Prop("low_charge", lambda s: s["charge"] == 0)
atB = Prop("atB", lambda s: s["loc"] == "B")
charged = Prop("charged", lambda s: s["charge"] > 0)

# Examples:
# - Eventually we hit low charge
# - Always, if low then eventually charged
formula1 = EF(low)
formula2 = AG(Implies(low, EF(charged)))

results = {
    "EF low_charge": eval_formula(formula1, m),
    "AG (low -> EF charged)": eval_formula(formula2, m)
}

for k, v in results.items():
    print(f"{k}: holds in states {sorted(v)}")
