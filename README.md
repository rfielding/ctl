
# POBTL* Model Checker and LTL Evaluator

This project contains a hybrid temporal logic framework that supports:

- ✅ Branching-time modal logic (PCTL*/CTL*)
- ✅ Linear-time temporal logic (LTL)
- ✅ Past-time operators (POBTL*)
- ✅ Strong implication (`StrongImplies`)
- ✅ Trace-based and model-based reasoning

---

## 🧠 Files

### `pobtl_model_checker.py`
- Implements CTL*/PCTL* logic with forward (`EF`, `AF`, `EG`, `AG`) and backward (`EP`, `AP`, `EH`, `AH`) operators
- Supports strong implication (`StrongImplies`)
- Provides a Kripke structure evaluator with recursive fixpoint semantics

### `ltl_trace.py`
- Evaluates LTL and past-LTL logic over linear traces
- Operators supported: `X`, `F`, `G`, `U`, `P`, `H`, `S`
- Evaluation via `eval_ltl(formula, trace, idx)`

### `electric_car_model.py`
- Defines a simple Kripke model of an electric car moving and charging
- Demonstrates modal properties like `EF(low_charge)` and `AG(low → EF(charged))`

### `ltl_examples.md`
- A cheat sheet of LTL + past-LTL formulas
- Example trace and usage examples with code snippets

---

## ✅ Getting Started

### Run POBTL* Example

```bash
python3 electric_car_model.py
```

### Evaluate LTL Formulas

```python
from ltl_trace import *

trace = [
    {"id": 0, "charged": False},
    {"id": 1, "charged": False},
    {"id": 2, "charged": True},
    {"id": 3, "charged": True},
    {"id": 4, "charged": False}
]

charged = Prop("charged", lambda s: s["charged"])
result = eval_ltl(F(charged), trace)
print(result)
```

---

## 🧪 Supported Modalities

| Class | Meaning |
|-------|---------|
| `EF(p)` | possibly eventually p |
| `AF(p)` | necessarily eventually p |
| `EG(p)` | possibly always p |
| `AG(p)` | necessarily always p |
| `EP(p)` | possibly once p |
| `AP(p)` | necessarily once p |
| `EH(p)` | possibly always past p |
| `AH(p)` | necessarily always past p |
| `F(p)` | eventually (future) |
| `G(p)` | always (future) |
| `P(p)` | once (past) |
| `H(p)` | always (past) |
| `U(p, q)` | p until q |
| `S(p, q)` | p since q |
| `StrongImplies(p, q)` | p implies q when p holds |

---

## 📂 Suggested Usage

- Combine LTL and CTL* to check safety, liveness, and recovery
- Use past operators for runtime log auditing
- Explore trace execution vs. Kripke graph reasoning

