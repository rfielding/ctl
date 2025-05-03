
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
