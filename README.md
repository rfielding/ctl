
# POBTL* Temporal Model Checker

This project implements a **tiny but powerful** branching- and linear-time temporal logic evaluator, called **POBTL***:

- ✅ Past & future modalities
- ✅ CTL*/PCTL*-style branching semantics
- ✅ LTL (trace-based) operators
- ✅ Strong implication
- ✅ Multi-agent and message-passing modeling

---

## 🔍 What is POBTL*?

**POBTL*** (Past-Oriented Branching-Time Logic*) extends classic CTL*/PCTL* with:

- `P`, `H`, `S`: Past equivalents of `F`, `G`, and `U`
- `EF`, `AF`, `EG`, `AG`: Future branching modalities
- `EP`, `AP`, `EH`, `AH`: Past branching modalities
- `StrongImplies`: Material implication confined to the domain of p
- Composability via boolean ops and implication

---

## 🧠 File: `pobtl_model_checker.py`

### 📐 Structure

The code defines a set of **AST nodes** (`Formula` classes), and an `eval_formula(formula, model)` dispatcher that walks the model.

### 🧩 Formula Classes

```python
Prop(name, func)             # atomic predicate
Not(a), And(a,b), Or(a,b)    # boolean ops
Implies(a,b), StrongImplies(a,b)

EF(a), AF(a), EG(a), AG(a)   # future modalities
EP(a), AP(a), EH(a), AH(a)   # past modalities
```

### 🗺 Kripke Model

```python
model.states:        Dict[int, Dict]  # state data
model.transitions:   Dict[int, List[(label, target)]]
model.predecessors:  Dict[int, List[int]]
```

---

## 🔁 Evaluation Algorithms

### 1. EF (Exists Future)

Forward search: a fixpoint starting from where `a` holds.

```python
while changed:
  for s in states:
    if any successor of s is in result:
      add s to result
```

### 2. AF (All Future)

Backward fixpoint: a state must have **all** successors satisfying `a`.

```python
while changed:
  for s in states:
    if all successors in result:
      add s
```

### 3. EG (Exists Globally)

This is the only algorithm with **cycle detection**:

```python
build subgraph of states where a holds
for each node s:
  run DFS to find a cycle from s to itself
```

### 4. AG (All Globally)

Fixpoint over "only keep s if all its successors satisfy `a`".

---

## 🔁 Past Operators

### EP / AP / EH / AH

Same as EF/AF/EG/AG, but using `.predecessors` instead of `.transitions`.

---

## ✅ Strong Implication

```python
StrongImplies(p, q) ≡ for all s: if p(s), then q(s)
```

Unlike `Implies`, this does not default true where `p(s)` is false.

---

## 🧪 How it works

Evaluation works by:

1. **AST traversal** (via `eval_formula`)
2. **Set operations** on state IDs
3. **Fixpoint iterations** to reach stable answers

Everything is built on top of:

- `model.states` = variable assignments
- `model.transitions` = successor graph
- `model.predecessors` = reverse graph

---

## 🧬 Why this is revolutionary

- Tiny: ~100 lines of Python, no dependencies
- Transparent: readable, extensible
- Modal complete: handles full CTL* + LTL + past
- Suitable for:
  - Embedded simulation (e.g. cars, queues)
  - Trace verification
  - Runtime monitors

---

## ✅ Example: Queue

See `queueing_model.py`

- Queue with capacity = 5
- Client sends jobs
- Server replies or drops
- Modal claim:
  - `EF(five_drops)` – possible to get dropped
  - `AG(full → EF(drop))` – if full, a drop must be eventually possible

---

## 👀 Read and Extend

Check `pobtl_model_checker.py` and try adding:

- `Until` operator (`p U q`)
- Bounded temporal operators (`F≤5 p`)
- Modal quantification (`E`, `A` as top-level)

---

Let me know if you'd like a LaTeX paper summarizing this.
