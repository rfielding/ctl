
# POBTL* Model Checker

This is a tiny yet expressive Python implementation of a **modal logic model checker** supporting:

- ✅ CTL*: `EF`, `AF`, `EG`, `AG`
- ✅ Past modalities: `EP`, `AP`, `EH`, `AH`
- ✅ Strong implication: `EF(p) ∧ AG(p → q)`
- ✅ Boolean logic: `∧`, `∨`, `¬`, `→`, `↔` (as syntactic sugar)

---

## 🔍 Modal Operators

| Operator | Name                  | Meaning                                                                 |
|----------|-----------------------|-------------------------------------------------------------------------|
| `EF φ`   | Exists Finally        | φ is possibly reachable in the future (`◇φ`)                            |
| `AF φ`   | Always Finally        | On all paths, φ eventually holds                                        |
| `EG φ`   | Exists Globally       | There exists a path where φ holds forever (including cycles)            |
| `AG φ`   | Always Globally       | On all paths, φ always holds (`□φ`)                                     |
| `EP φ`   | Exists Previously     | φ possibly held at some past point                                      |
| `AP φ`   | Always Previously     | φ always held in all past paths                                         |
| `EH φ`   | Exists Historically   | There is a cycle in the past where φ holds                              |
| `AH φ`   | Always Historically   | φ always held along all past paths (□ in reverse)                       |

---

## ↔️ Strong Implication

Defined as:

```
StrongImplies(p, q) ≡ EF(p) ∧ AG(p → q)
```

This checks that:

1. `p` is possibly reachable
2. Whenever `p` is true, `q` always follows

---

## ↔️ Bi-implication

Defined as syntactic sugar:

```
Iff(p, q) ≡ (p → q) ∧ (q → p)
```

And used in modal contexts like:

- `AG(Iff(p, q))` ≡ `AG((p → q) ∧ (q → p))`

---

## ✅ Example

```python
P = Prop("P", lambda s: s["x"] > 0)
Q = Prop("Q", lambda s: s["y"] < 5)

formula = AG(Implies(P, Q))
result_states = eval_formula(formula, model)
```

---

## 📁 Files

- `pobtl_model_checker.py`: The logic engine
- `tests.py`: Runs propositional and modal logic assertions


---

## 🔧 Running with `uv` and Managing Permissions

We recommend using [`uv`](https://github.com/astral-sh/uv) for installing dependencies efficiently:

```bash
uv pip install -r requirements.txt
```

This will ensure all Python packages are installed in your environment using `uv`, which is fast and compatible with Python 3.12.7.

### 🔒 Permissions Note

If you're running on a Unix-based system and you downloaded these scripts via browser or extracted them from a zip archive, they may not be executable by default.

You can fix permissions like this:

```bash
chmod +x chatbot_requirements_logger.py
```

Or run directly with:

```bash
python3 chatbot_requirements_logger.py
```

Make sure your environment has:

- `OPENAI_API_KEY` set:  
  ```bash
  export OPENAI_API_KEY=sk-...
  ```

- Optional: set your active project  
  ```bash
  export PROJECT=myproject
  ```

Then you're ready to go.

---

---

## 🧠 POBTL* Syntax Quick Reference

This section shows how to write modal logic formulas in valid Python using the `pobtl_model_checker.py` module.

### 🧩 Define Propositions

Use `Prop(name, lambda state: condition)` to define atomic logic:

```python
from pobtl_model_checker import Prop, EF, eval_formula

# Proposition: queue is full
q = Prop("q", lambda s: s["Queue"] == 5)
```

### ⏳ Use Modal Operators

All operators like `EF`, `AF`, `AG`, `EG`, `AP`, `EP`, `AH`, `EH`, `StrongImplies`, etc. are Python classes that operate on `Prop` or logical expressions.

```python
check_first_drop = EF(q)
```

### ✅ Evaluate Against States

To compute which states satisfy a formula:

```python
states = all_possible_states()  # You must define this
result_states = eval_formula(check_first_drop, states)
probability = len(result_states) / len(states)
```

### ❗ Requirements

- All POBTL* formulas are just Python objects.
- You must provide a list of state dictionaries: `[{"Queue": 0}, {"Queue": 1}, ..., {"Queue": 5}]`
- Do **not** invent new DSLs or use symbolic variables — use Python.

---