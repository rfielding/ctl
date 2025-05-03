
# queueing_model.py

import random
from pobtl_model_checker import *

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

m = Model()

# Parameters
CAPACITY = 5
arrival_rate = 0.2
departure_rate = 0.3

# States: simulate a small sequence of job arrivals and departures
states = []
for queue_len in range(CAPACITY + 1):
    for client_drops in range(4):  # client drop counter
        s = m.add_state({
            "queue": queue_len,
            "client_drop_count": client_drops,
            "msg": None,
        })
        states.append((queue_len, client_drops, s))

# Add transitions (arrival, drop, departure)
for queue_len, client_drops, sid in states:
    if queue_len < CAPACITY:
        # Normal arrival
        next_state = next(s for q, d, s in states if q == queue_len + 1 and d == client_drops)
        m.add_transition(sid, next_state, label="arrival")
    else:
        # Drop: full queue, increment drop count, msg = 'drop'
        if client_drops < 3:
            next_state = next(s for q, d, s in states if q == queue_len and d == client_drops + 1)
            m.add_transition(sid, next_state, label="drop")

    if queue_len > 0:
        # Departure, msg = 'result'
        next_state = next(s for q, d, s in states if q == queue_len - 1 and d == client_drops)
        m.add_transition(sid, next_state, label="departure")

# Modal properties
drop = Prop("drop", lambda s: s["client_drop_count"] > 0)
five_drops = Prop("five_drops", lambda s: s["client_drop_count"] >= 3)
queue_full = Prop("full", lambda s: s["queue"] == CAPACITY)

fml1 = EF(five_drops)
fml2 = AG(Implies(queue_full, EF(drop)))

print("EF five_drops:", sorted(eval_formula(fml1, m)))
print("AG (full -> EF drop):", sorted(eval_formula(fml2, m)))
