# example-REQUIREMENTS.md

This file contains modal logic examples embedded in code fences using POBTL* syntax.

## ✅ Check if queue can reach full (Queue == 5)

```pobtl
#!/bin/env python3

from pobtl_model_checker import Prop, EF, Model, eval_formula

# Define simple state space
states = [ {"Queue": i} for i in range(6) ]
model = Model(states)

# Proposition: queue is full
q = Prop("q", lambda s: s["Queue"] == 5)
check_full = EF(q)

# Evaluate formula
matching = eval_formula(check_full, model)
print("✅ States where EF(Queue == 5) holds:", [dict(s) for s in matching])
```

More tests can be added below...