
# pobtl_model_checker.py
# ---------------------------------------------
# A minimal temporal model checker for POBTL*
# Supports: EF, AF, EG, AG, EP, AP, EH, AH, AND bi-implication
# ---------------------------------------------

class Formula: pass

class Prop(Formula):
    def __init__(self, name, func):
        self.name = name
        self.func = func

class Not(Formula):
    def __init__(self, sub):
        self.sub = sub

class And(Formula):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Or(Formula):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Implies(Formula):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Iff(Formula):  # Bi-implication: (a <-> b) ≡ (a → b) ∧ (b → a)
    def __init__(self, a, b):
        self.a = a
        self.b = b

class StrongImplies(Formula):  # EF(a) ∧ AG(a → b)
    def __init__(self, a, b):
        self.a = a
        self.b = b

class EF(Formula):  # Eventually Future
    def __init__(self, sub): self.sub = sub

class AF(Formula):  # Always Eventually Future
    def __init__(self, sub): self.sub = sub

class EG(Formula):  # Exists Globally Future
    def __init__(self, sub): self.sub = sub

class AG(Formula):  # All Globally Future
    def __init__(self, sub): self.sub = sub

class EP(Formula):  # Exists Previously
    def __init__(self, sub): self.sub = sub

class AP(Formula):  # All Previously
    def __init__(self, sub): self.sub = sub

class EH(Formula):  # Exists Historically
    def __init__(self, sub): self.sub = sub

class AH(Formula):  # All Historically
    def __init__(self, sub): self.sub = sub

def eval_EG(formula, model):
    sat = eval_formula(formula.sub, model)
    graph = {s: [t for _, t in model.transitions[s] if t in sat] for s in sat}
    visited = set()
    def dfs(u, origin):
        visited.add(u)
        for v in graph.get(u, []):
            if v == origin or (v not in visited and dfs(v, origin)):
                return True
        return False
    return {s for s in sat if dfs(s, s)}

def eval_formula(formula, model):
    if isinstance(formula, Iff):
        return eval_formula(And(Implies(formula.a, formula.b), Implies(formula.b, formula.a)), model)
    if isinstance(formula, StrongImplies):
        return eval_formula(And(EF(formula.a), AG(Implies(formula.a, formula.b))), model)
    if isinstance(formula, Prop):
        return {s for s, state in model.states.items() if formula.func(state)}
    if isinstance(formula, Not):
        return set(model.states) - eval_formula(formula.sub, model)
    if isinstance(formula, And):
        return eval_formula(formula.a, model) & eval_formula(formula.b, model)
    if isinstance(formula, Or):
        return eval_formula(formula.a, model) | eval_formula(formula.b, model)
    if isinstance(formula, Implies):
        return eval_formula(Or(Not(formula.a), formula.b), model)
    if isinstance(formula, EF):
        sat = eval_formula(formula.sub, model)
        result = set(sat)
        changed = True
        while changed:
            changed = False
            for s, edges in model.transitions.items():
                if any(t in result for _, t in edges) and s not in result:
                    result.add(s)
                    changed = True
        return result
    if isinstance(formula, AF):
        sat = eval_formula(formula.sub, model)
        result = set(sat)
        changed = True
        while changed:
            changed = False
            for s, edges in model.transitions.items():
                if all(t in result for _, t in edges) and s not in result:
                    result.add(s)
                    changed = True
        return result
    if isinstance(formula, AG):
        result = eval_formula(formula.sub, model)
        changed = True
        while changed:
            changed = False
            for s, edges in model.transitions.items():
                if all(t in result for _, t in edges) and s not in result:
                    result.add(s)
                    changed = True
        return result
    if isinstance(formula, EG):
        return eval_EG(formula, model)
    if isinstance(formula, EP):
        sat = eval_formula(formula.sub, model)
        result = set(sat)
        changed = True
        while changed:
            changed = False
            for s, preds in model.predecessors.items():
                if any(p in result for p in preds) and s not in result:
                    result.add(s)
                    changed = True
        return result
    if isinstance(formula, AP):
        sat = eval_formula(formula.sub, model)
        result = set(sat)
        changed = True
        while changed:
            changed = False
            for s, preds in model.predecessors.items():
                if preds and all(p in result for p in preds) and s not in result:
                    result.add(s)
                    changed = True
        return result
    if isinstance(formula, EH):
        sat = eval_formula(formula.sub, model)
        graph = {s: [p for p in model.predecessors[s] if p in sat] for s in sat}
        visited = set()
        def dfs(u, origin):
            visited.add(u)
            for v in graph.get(u, []):
                if v == origin or (v not in visited and dfs(v, origin)):
                    return True
            return False
        return {s for s in sat if dfs(s, s)}
    if isinstance(formula, AH):
        result = eval_formula(formula.sub, model)
        changed = True
        while changed:
            changed = False
            for s, preds in model.predecessors.items():
                if preds and all(p in result for p in preds) and s not in result:
                    result.add(s)
                    changed = True
        return result
    return set()
