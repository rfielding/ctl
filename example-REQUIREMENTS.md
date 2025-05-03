## Assistant @ 2025-05-03T14:42:00

Here's a valid POBTL* block for testing:

```pobtl
from pobtl_model_checker import Prop, EF, eval_formula

states = [ {"Queue": i} for i in range(6) ]

q = Prop("q", lambda s: s["Queue"] == 5)
check_full = EF(q)

matching = eval_formula(check_full, states)
print("✅ States where EF(Queue == 5) holds:", matching)
```