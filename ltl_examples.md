
# LTL + Past-LTL Formula Examples

## Trace
```python
trace = [
    {"id": 0, "charged": False},
    {"id": 1, "charged": False},
    {"id": 2, "charged": True},
    {"id": 3, "charged": True},
    {"id": 4, "charged": False}
]
```

## Formulas
```python
charged = Prop("charged", lambda s: s["charged"])
not_charged = Not(charged)
```

### Future Temporal Logic
| Formula | Description |
|---------|-------------|
| `F(charged)` | Eventually charged |
| `G(charged)` | Always charged |
| `X(charged)` | Next state charged |
| `charged U not_charged` | Charged until not_charged |

### Past Temporal Logic
| Formula | Description |
|---------|-------------|
| `P(charged)` | Charged was true at some point |
| `H(charged)` | Charged has always been true |
| `charged S not_charged` | Charged has held since not_charged |

### Strong Implication
| Formula | Description |
|---------|-------------|
| `StrongImplies(charged, not_charged)` | Whenever charged, then also not_charged |
| `F(StrongImplies(charged, not_charged))` | Eventually, charged → not_charged |
| `G(StrongImplies(charged, charged))` | Always, charged implies itself |

### Logical Combinations
```python
Not(F(charged))              
And(F(charged), G(charged)) 
Or(P(charged), H(not_charged))
Implies(G(charged), F(not_charged))
StrongImplies(P(charged), charged)
```

### Evaluation
```python
from ltl_trace import *
result = eval_ltl(F(charged), trace)
print("F(charged) =", result)
```
