#!/bin/env python3

# POBTL* Model Checker Core

from typing import Any, Callable, Dict, List, Union
from dataclasses import dataclass

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

@dataclass 
class Y:  # Yesterday/Previous
    f: Any
    def eval(self, model, state):
        # Find all states that could transition to current
        current = frozenset(state.items()) if isinstance(state, dict) else state
        possible_pasts = set()
        for s in model.states:
            s_frozen = frozenset(s.items())
            if current in model.transitions[s_frozen]:
                possible_pasts.add(s_frozen)
        return any(self.f.eval(model, dict(s)) for s in possible_pasts)

@dataclass
class O:  # Once (in the past)
    f: Any
    def eval(self, model, state):
        visited = set()
        def check_past(s):
            s_frozen = frozenset(s.items()) if isinstance(s, dict) else s
            if s_frozen in visited:
                return False
            visited.add(s_frozen)
            if self.f.eval(model, dict(s_frozen)):
                return True
            # Check all possible previous states
            for prev_s in model.states:
                prev_frozen = frozenset(prev_s.items())
                if s_frozen in model.transitions[prev_frozen]:
                    if check_past(prev_s):
                        return True
            return False
        return check_past(state)

@dataclass
class H:  # Historically (always in the past)
    f: Any
    def eval(self, model, state):
        visited = set()
        def check_past(s):
            # Convert state to frozen set for consistent handling
            s_frozen = frozenset(s.items()) if isinstance(s, dict) else s
            
            # If we've seen this state before in our traversal:
            # Return True because we've already verified this path
            # (and we want to handle cycles gracefully)
            if s_frozen in visited:
                return True
            
            # Mark this state as visited to handle cycles
            visited.add(s_frozen)
            
            # FIRST: Check if property holds in current state
            # If it doesn't, we can fail fast - history is violated
            current_state = dict(s_frozen)
            if not self.f.eval(model, current_state):
                return False
            
            # SECOND: Find all states that could lead to current state
            # by checking which states have transitions to us
            predecessor_states = []
            for prev_s in model.states:
                prev_frozen = frozenset(prev_s.items())
                if s_frozen in model.transitions[prev_frozen]:
                    predecessor_states.append(prev_frozen)
            
            # THIRD: Handle different cases:
            
            # Case 1: No predecessors (we're at an initial state)
            # If we got here, current state satisfies f, and that's all we need
            if not predecessor_states:
                return True
            
            # Case 2: Has predecessors
            # Property must hold in ALL predecessor paths
            for prev_state in predecessor_states:
                if not check_past(prev_state):
                    return False
            
            # If we get here, property held in current state
            # and recursively in all possible past paths
            return True
            
        # Start checking from the given state
        return check_past(state)

@dataclass
class S:  # Since
    f1: Any  # First formula
    f2: Any  # Second formula
    def eval(self, model, state):
        visited = set()
        def check_since(s):
            s_frozen = frozenset(s.items()) if isinstance(s, dict) else s
            if s_frozen in visited:
                return False
            visited.add(s_frozen)
            # f2 holds now
            if self.f2.eval(model, dict(s_frozen)):
                return True
            # f1 holds now and since holds in some previous state
            if self.f1.eval(model, dict(s_frozen)):
                for prev_s in model.states:
                    prev_frozen = frozenset(prev_s.items())
                    if s_frozen in model.transitions[prev_frozen]:
                        if check_since(prev_s):
                            return True
            return False
        return check_since(state)