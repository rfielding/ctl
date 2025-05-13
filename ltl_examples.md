# LTL + Past-LTL Formula Examples

## Trace Example
```python
trace = [
    {"state": "initial"},
    {"state": "waiting_for_input"},
    {"state": "processing"},
    {"state": "responding"},
    {"state": "waiting_for_input"}
]
```

## Basic Propositions
```python
# Define basic state checks
is_processing = Prop("processing", lambda s: s["state"] == "processing")
is_waiting = Prop("waiting", lambda s: s["state"] == "waiting_for_input")
is_responding = Prop("responding", lambda s: s["state"] == "responding")
```

### Future Temporal Logic (Looking Forward)
| Operator | Formula Example | Description |
|----------|----------------|-------------|
| `F` (Finally) | `F(is_processing)` | Eventually the system will be processing |
| `G` (Globally) | `G(is_waiting)` | From now on, system will always be waiting |
| `X` (Next) | `X(is_responding)` | In the next state, system will be responding |
| `U` (Until) | `is_waiting U is_processing` | System waits until processing begins |

### Past Temporal Logic (Looking Backward)
| Operator | Formula Example | Description |
|----------|----------------|-------------|
| `O` or `P` (Once/Past) | `O(is_processing)` | At some point in the past, system was processing |
| `H` (Historically) | `H(is_waiting)` | System has always been in waiting state |
| `Y` (Yesterday) | `Y(is_responding)` | In the previous state, system was responding |
| `S` (Since) | `is_waiting S is_processing` | System has been waiting since it was processing |

### Common Requirement Patterns

1. Response Property:
```python
G(is_waiting → F(is_responding))  # Every wait state is eventually followed by a response
```

2. Precedence Property:
```python
G(is_responding → O(is_processing))  # Every response must be preceded by processing
```

3. Bounded Response:
```python
G(is_waiting → X(F(is_responding)))  # Response happens within one step of waiting
```

4. State Sequence:
```python
G(is_processing → (X(is_responding) ∧ O(is_waiting)))  # Processing must be between waiting and responding
```

### Evaluation Example
```python
from ltl_trace import *

# Check if a response always follows processing
formula = G(StrongImplies(is_processing, F(is_responding)))
result = eval_ltl(formula, trace)
print("Property satisfied:", result)
```

## Common Use Cases

1. Protocol Verification:
   - `G(request → F(response))` - Every request eventually gets a response
   - `G(response → O(request))` - Every response must have had a request

2. Safety Properties:
   - `G(not_error)` - System never enters error state
   - `H(valid_state)` - System has always been in a valid state

3. Liveness Properties:
   - `G(F(progress))` - System always eventually makes progress
   - `F(G(stable))` - System eventually becomes permanently stable

4. Fairness Properties:
   - `G(waiting → F(served))` - Every waiting client is eventually served
   - `G(F(reset))` - System is reset infinitely often

## Notes
- Future operators (F, G, X, U) look forward from the current state
- Past operators (O/P, H, Y, S) look backward from the current state
- Combining past and future operators allows expressing complex temporal relationships
- The trace evaluation starts at index 0 and moves forward
