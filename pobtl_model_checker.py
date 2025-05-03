# pobtl_model_checker.py
# Multi-modal branching-time temporal logic evaluator (POBTL*)

class Formula: pass

class Prop(Formula):       def __init__(self, name, func): self.name, self.func = name, func
class Not(Formula):        def __init__(self, sub): self.sub = sub
class And(Formula):        def __init__(self, a, b): self.a, self.b = a, b
class Or(Formula):         def __init__(self, a, b): self.a, self.b = a, b
class Implies(Formula):    def __init__(self, a, b): self.a, self.b = a, b
class StrongImplies(Formula): def __init__(self, a, b): self.a, self.b = a, b
class EF(Formula):         def __init__(self, sub): self.sub = sub
class AF(Formula):         def __init__(self, sub): self.sub = sub
class EG(Formula):         def __init__(self, sub): self.sub = sub
class AG(Formula):         def __init__(self, sub): self.sub = sub
class EP(Formula):         def __init__(self, sub): self.sub = sub
class AP(Formula):         def __init__(self, sub): self.sub = sub
class EH(Formula):         def __init__(self, sub): self.sub = sub
class AH(Formula):         def __init__(self, sub): self.sub = sub

def eval_EG(formula, model):
    sat = eval_formula(formula.sub, model)
    graph = {s: [t for _, t in model.transitions[s] if t in sat] for s in sat}
    visited, stack = set(), []
    def dfs(u, origin):
        visited.add(u)
        stack.append(u)
        for v in graph.get(u, []):
            if v == origin or (v not in visited and dfs(v, origin)):
                return True
        stack.pop()
        return False
    return {s for s in sat if dfs(s, s)}

def eval_formula(formula, model):
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
    if isinstance(formula, StrongImplies):
        a = eval_formula(formula.a, model)
        b = eval_formula(formula.b, model)
        return {s for s in model.states if s not in a or s in b}
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
        visited, stack = set(), []
        def dfs(u, origin):
            visited.add(u)
            stack.append(u)
            for v in graph.get(u, []):
                if v == origin or (v not in visited and dfs(v, origin)):
                    return True
            stack.pop()
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
