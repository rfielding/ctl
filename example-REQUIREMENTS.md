```pobtl
from pobtl_model_checker import Prop, EF, Model, eval_formula

states = [{"Queue": i} for i in range(6)]
transitions = {
    frozenset(s.items()): [
        frozenset(t.items()) for t in states
        if abs(s["Queue"] - t["Queue"]) <= 1
    ]
    for s in states
}

q = Prop("q", lambda s: s["Queue"] == 5)
check_full = EF(q)

model = Model(states, transitions)
matching = eval_formula(check_full, model)
print("✅ States where EF(Queue == 5) holds:", matching)
```

