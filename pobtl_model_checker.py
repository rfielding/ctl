
# pobtl_model_checker.py
# ---------------------------------------------
# A minimal temporal model checker for CTL*, PCTL*, and past/future modalities.
# This includes support for:
#   - EF, AF, EG, AG: Future temporal modalities
#   - EP, AP, EH, AH: Past temporal modalities
#   - Strong implication: EF(p) ∧ AG(p → q)
#   - Boolean operations: And, Or, Not, Implies
# ---------------------------------------------

# Base class for all formula AST nodes
class Formula: pass

# Atomic proposition: test on state data
class Prop(Formula):
    def __init__(self, name, func):
        self.name = name  # string name
        self.func = func  # lambda: state → bool

# Boolean operators
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

# Strong implication: EF(a) ∧ AG(a → b)
class StrongImplies(Formula):
    def __init__(self, a, b):
        self.a = a
        self.b = b

# Future-oriented modalities
class EF(Formula):  # Eventually (Exists path)
    def __init__(self, sub): self.sub = sub

class AF(Formula):  # Eventually (All paths)
    def __init__(self, sub): self.sub = sub

class EG(Formula):  # Always (Exists path, in some cycle)
    def __init__(self, sub): self.sub = sub

class AG(Formula):  # Always (All paths)
    def __init__(self, sub): self.sub = sub

# Past-oriented modalities
class EP(Formula):  # Previously Eventually (Exists past path)
    def __init__(self, sub): self.sub = sub

class AP(Formula):  # Previously Always (All past paths)
    def __init__(self, sub): self.sub = sub

class EH(Formula):  # Historically Always (Exists path in a cycle backwards)
    def __init__(self, sub): self.sub = sub

class AH(Formula):  # Historically Always (All paths)
    def __init__(self, sub): self.sub = sub

# Cycle detection logic used for EG and EH
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

# Main evaluation dispatch function
def eval_formula(formula, model):
    # Desugar strong implication
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
        # EF: fixpoint of "can reach a satisfying state"
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
        # AF: fixpoint where *all successors* reach `a`
        sat = eval_formula(formula.sub, model)
        result = set(sat)
        changed = True
        while changed:
            changed = False
            for s, edges in model.transitions.items():
                if edges and all(t in result for _, t in edges) and s not in result:
                    result.add(s)
                    changed = True
        return result

    if isinstance(formula, AG):
        # AG: fixpoint where all successors already satisfy `a`
        result = eval_formula(formula.sub, model)
        changed = True
        while changed:
            changed = False
            for s, edges in model.transitions.items():
                if edges and all(t in result for _, t in edges) and s not in result:
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
