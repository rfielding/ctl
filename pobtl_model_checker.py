#!/bin/env python3

# POBTL* Model Checker Core

from typing import Any, Callable, Dict, List, Union, Set, FrozenSet, Tuple
from dataclasses import dataclass

State = Dict[str, Any]
StateItems = FrozenSet[tuple[str, Any]]

# === Formula Base ===

class Formula:
    def eval(self, model: 'Model', state: State) -> bool:
        raise NotImplementedError()

# === Propositions ===

class Prop(Formula):
    def __init__(self, name: str, func: Callable[[State], bool]):
        self.name = name
        self.func = func
    def eval(self, model: 'Model', state: State) -> bool:
        return self.func(state)
    def __repr__(self): return self.name

# === Boolean Connectives ===

class Not(Formula):
    def __init__(self, f: Formula):
        self.f = f
    def eval(self, model: 'Model', state: State) -> bool:
        return not self.f.eval(model, state)

class And(Formula):
    def __init__(self, f: Formula, g: Formula):
        self.f, self.g = f, g
    def eval(self, model: 'Model', state: State) -> bool:
        return self.f.eval(model, state) and self.g.eval(model, state)

class Or(Formula):
    def __init__(self, f: Formula, g: Formula):
        self.f, self.g = f, g
    def eval(self, model: 'Model', state: State) -> bool:
        return self.f.eval(model, state) or self.g.eval(model, state)

class Implies(Formula):
    def __init__(self, f: Formula, g: Formula):
        self.f, self.g = f, g
    def eval(self, model: 'Model', state: State) -> bool:
        return (not self.f.eval(model, state)) or self.g.eval(model, state)

class Iff(Formula):
    def __init__(self, f, g): self.f, self.g = f, g
    def eval(self, model, state): return self.f.eval(model, state) == self.g.eval(model, state)

def StrongImplies(p, q): return And(EF(p), AG(Implies(p, q)))

# === Modal Operators ===

class EF(Formula):
    def __init__(self, f: Formula):
        self.f = f
    def eval(self, model: 'Model', state: State) -> bool:
        return model.reachable(state, self.f)

class AF(Formula):
    def __init__(self, f: Formula):
        self.f = f
    def eval(self, model: 'Model', state: State) -> bool:
        return model.must_reach(state, self.f)

class EG(Formula):
    def __init__(self, f: Formula):
        self.f = f
    def eval(self, model: 'Model', state: State) -> bool:
        return model.exists_globally(state, self.f)

class AG(Formula):
    def __init__(self, f: Formula):
        self.f = f
    def eval(self, model: 'Model', state: State) -> bool:
        return model.always_globally(state, self.f)

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

def hashable(state: State) -> StateItems:
    return frozenset(state.items())

class Model:
    def __init__(self, states: List[State], transitions: Dict[StateItems, List[StateItems]]):
        self.states = states
        self.transitions = transitions
        # Cache for predecessors
        self._predecessor_cache: Dict[StateItems, List[State]] = {}
        # Cache for hashable states
        self._hashable_cache: Dict[str, StateItems] = {}

    def get_transitions(self, state: State) -> List[State]:
        state_items = hashable(state)
        if state_items not in self.transitions:
            return []
        return [dict(t) for t in self.transitions[state_items]]

    def get_predecessors(self, state: State) -> List[State]:
        state_items = hashable(state)
        if state_items in self._predecessor_cache:
            return self._predecessor_cache[state_items]
            
        result = []
        for src_items, targets in self.transitions.items():
            if state_items in targets:
                result.append(dict(src_items))
        self._predecessor_cache[state_items] = result
        return result

    def reachable(self, state: State, f: Formula) -> bool:
        visited: Set[StateItems] = set()
        # Use deque instead of list for stack (faster pop/append)
        from collections import deque
        stack = deque([self._get_hashable(state)])
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                current_dict = dict(current)
                # Check formula first (short circuit)
                if f.eval(self, current_dict):
                    return True
                # Add unvisited successors
                stack.extend(
                    next_state for next_state in self.transitions.get(current, [])
                    if next_state not in visited
                )
        return False

    def must_reach(self, state: State, f: Formula) -> bool:
        visited: Set[StateItems] = set()
        def dfs(s: State) -> bool:
            s_items = hashable(s)
            if s_items in visited:
                return True
            visited.add(s_items)
            if f.eval(self, s):
                return True
            next_states = self.transitions.get(s_items, [])
            return next_states and all(dfs(dict(t)) for t in next_states)
        return dfs(state)

    def exists_globally(self, state: State, f: Formula) -> bool:
        path: List[StateItems] = []
        visited: Set[StateItems] = set()
        def dfs(s: State) -> bool:
            s_items = hashable(s)
            if s_items in path:
                return True
            if s_items in visited:
                return False
            visited.add(s_items)
            if not f.eval(self, s):
                return False
            path.append(s_items)
            for next_state in self.transitions.get(s_items, []):
                if dfs(dict(next_state)):
                    return True
            path.pop()
            return False
        return dfs(state)

    def always_globally(self, state: State, f: Formula) -> bool:
        visited: Set[StateItems] = set()
        def dfs(s: State) -> bool:
            s_items = hashable(s)
            if s_items in visited:
                return True
            visited.add(s_items)
            if not f.eval(self, s):
                return False
            return all(dfs(dict(t)) for t in self.transitions.get(s_items, []))
        return dfs(state)

    def reachable_past(self, state, f):
        return self.reachable(state, f)

    def must_reach_past(self, state, f):
        return self.must_reach(state, f)

    def exists_globally_past(self, state, f):
        return self.exists_globally(state, f)

    def always_globally_past(self, state, f):
        return self.always_globally(state, f)

    # Add helper method to cache hashable states
    def _get_hashable(self, state: State) -> StateItems:
        # Use string representation as dict key
        key = str(sorted(state.items()))
        if key not in self._hashable_cache:
            self._hashable_cache[key] = hashable(state)
        return self._hashable_cache[key]

# === Entry point ===

def eval_formula(formula: Formula, model_or_states: Union[Model, List[Dict]]) -> List[Dict]:
    if isinstance(model_or_states, list):
        raise ValueError("Must wrap state list in a Model before evaluating.")
    model = model_or_states
    return [s for s in model.states if formula.eval(model, s)]

@dataclass 
class Y(Formula):  # Yesterday/Previous
    f: Formula
    def eval(self, model: 'Model', state: State) -> bool:
        return any(self.f.eval(model, pred) for pred in model.get_predecessors(state))

@dataclass
class O(Formula):  # Once (in the past)
    f: Formula
    def eval(self, model: 'Model', state: State) -> bool:
        visited: Set[StateItems] = set()
        stack = [hashable(state)]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                if self.f.eval(model, dict(current)):
                    return True
                for pred in model.get_predecessors(dict(current)):
                    pred_items = hashable(pred)
                    if pred_items not in visited:
                        stack.append(pred_items)
        return False

@dataclass
class H(Formula):  # Historically (always in the past)
    f: Formula
    def eval(self, model: 'Model', state: State) -> bool:
        visited: Set[StateItems] = set()
        def check_past(s: State) -> bool:
            # Convert state to frozen set for consistent handling
            s_frozen = hashable(s)
            
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
            predecessor_states = model.get_predecessors(s)
            
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
class S(Formula):
    """Since operator: f1 has been true since f2 was true in the past"""
    def __init__(self, f1: Formula, f2: Formula):
        self.f1, self.f2 = f1, f2
    
    def eval(self, model: 'Model', state: State) -> bool:
        visited: Set[StateItems] = set()
        cache: Dict[StateItems, bool] = {}  # Cache results
        
        def check_since(s: State) -> bool:
            s_items = model._get_hashable(s)
            if s_items in cache:
                return cache[s_items]
            if s_items in visited:
                return False
                
            visited.add(s_items)
            
            # Check f2 first (short circuit)
            if self.f2.eval(model, s):
                cache[s_items] = True
                return True
                
            # Then check f1
            if not self.f1.eval(model, s):
                cache[s_items] = False
                return False
                
            # Check predecessors
            result = any(check_since(pred) for pred in model.get_predecessors(s))
            cache[s_items] = result
            return result
            
        return check_since(state)