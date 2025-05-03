#!/bin/env python3

# POBTL* Model Checker Core

from typing import Any, Callable, Dict, List, Union

# === Formula Base ===

class Formula:
    def eval(self, model, state): raise NotImplementedError()

# === Propositions ===

class Prop(Formula):
    def __init__(self, name: str, func: Callable[[Dict], bool]):
        self.name, self.func = name, func
    def eval(self, model, state): return self.func(state)
    def __repr__(self): return self.name

# === Boolean Connectives ===

class Not(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return not self.f.eval(model, state)

class And(Formula):
    def __init__(self, f, g): self.f, self.g = f, g
    def eval(self, model, state): return self.f.eval(model, state) and self.g.eval(model, state)

class Or(Formula):
    def __init__(self, f, g): self.f, self.g = f, g
    def eval(self, model, state): return self.f.eval(model, state) or self.g.eval(model, state)

class Implies(Formula):
    def __init__(self, f, g): self.f, self.g = f, g
    def eval(self, model, state): return (not self.f.eval(model, state)) or self.g.eval(model, state)

class Iff(Formula):
    def __init__(self, f, g): self.f, self.g = f, g
    def eval(self, model, state): return self.f.eval(model, state) == self.g.eval(model, state)

def StrongImplies(p, q): return And(EF(p), AG(Implies(p, q)))

# === Modal Operators ===

class EF(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.reachable(state, self.f)

class AF(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.must_reach(state, self.f)

class EG(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.exists_globally(state, self.f)

class AG(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.always_globally(state, self.f)

class EP(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.reachable_past(state, self.f)

class AP(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.must_reach_past(state, self.f)

class EH(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.exists_globally_past(state, self.f)

class AH(Formula):
    def __init__(self, f): self.f = f
    def eval(self, model, state): return model.always_globally_past(state, self.f)

# === Model ===

def hashable(state):
    return frozenset(state.items()) if isinstance(state, dict) else state

class Model:
    def __init__(self, states: List[Dict], transitions: Dict[Any, List[Any]]):
        self.states = states
        self.transitions = {
            hashable(s): [hashable(t) for t in targets]
            for s, targets in transitions.items()
        }

    def reachable(self, state, f):
        visited, stack = set(), [hashable(state)]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                sdict = dict(current)
                if f.eval(self, sdict): return True
                stack.extend(self.transitions.get(current, []))
        return False

    def must_reach(self, state, f):
        visited = set()
        def dfs(s):
            hs = hashable(s)
            if hs in visited: return True
            visited.add(hs)
            sdict = dict(hs)
            if f.eval(self, sdict): return True
            succs = self.transitions.get(hs, [])
            return succs and all(dfs(t) for t in succs)
        return dfs(state)

    def always_globally(self, state, f):
        visited = set()
        def dfs(s):
            hs = hashable(s)
            if hs in visited: return True
            visited.add(hs)
            sdict = dict(hs)
            if not f.eval(self, sdict): return False
            return all(dfs(t) for t in self.transitions.get(hs, []))
        return dfs(state)

    def exists_globally(self, state, f):
        path, visited = [], set()
        def dfs(s):
            hs = hashable(s)
            if hs in path: return True
            if hs in visited: return False
            visited.add(hs)
            sdict = dict(hs)
            if not f.eval(self, sdict): return False
            path.append(hs)
            for t in self.transitions.get(hs, []):
                if dfs(t): return True
            path.pop()
            return False
        return dfs(state)

    def reachable_past(self, state, f):
        return self.reachable(state, f)

    def must_reach_past(self, state, f):
        return self.must_reach(state, f)

    def exists_globally_past(self, state, f):
        return self.exists_globally(state, f)

    def always_globally_past(self, state, f):
        return self.always_globally(state, f)

# === Entry point ===

def eval_formula(formula: Formula, model_or_states: Union[Model, List[Dict]]) -> List[Dict]:
    if isinstance(model_or_states, list):
        raise ValueError("Must wrap state list in a Model before evaluating.")
    model = model_or_states
    return [s for s in model.states if formula.eval(model, s)]