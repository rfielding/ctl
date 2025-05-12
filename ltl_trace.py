
#!/usr/bin/env python3
# ltl_trace.py

class Formula:
    pass

class Prop(Formula):
    def __init__(self, name, func): self.name, self.func = name, func

class Not(Formula):
    def __init__(self, sub): self.sub = sub

class And(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class Or(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class Implies(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class StrongImplies(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class X(Formula):
    def __init__(self, sub): self.sub = sub

class F(Formula):
    def __init__(self, sub): self.sub = sub

class G(Formula):
    def __init__(self, sub): self.sub = sub

class U(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class S(Formula):
    def __init__(self, a, b): self.a, self.b = a, b

class P(Formula):
    def __init__(self, sub): self.sub = sub

class H(Formula):
    def __init__(self, sub): self.sub = sub

def eval_ltl(formula, trace, idx=0):
    if idx >= len(trace): return False
    s = trace[idx]
    if isinstance(formula, Prop): return formula.func(s)
    if isinstance(formula, Not): return not eval_ltl(formula.sub, trace, idx)
    if isinstance(formula, And): return eval_ltl(formula.a, trace, idx) and eval_ltl(formula.b, trace, idx)
    if isinstance(formula, Or): return eval_ltl(formula.a, trace, idx) or eval_ltl(formula.b, trace, idx)
    if isinstance(formula, Implies): return not eval_ltl(formula.a, trace, idx) or eval_ltl(formula.b, trace, idx)
    if isinstance(formula, StrongImplies): return not eval_ltl(formula.a, trace, idx) or eval_ltl(formula.b, trace, idx)
    if isinstance(formula, X): return idx+1 < len(trace) and eval_ltl(formula.sub, trace, idx+1)
    if isinstance(formula, F): return any(eval_ltl(formula.sub, trace, i) for i in range(idx, len(trace)))
    if isinstance(formula, G): return all(eval_ltl(formula.sub, trace, i) for i in range(idx, len(trace)))
    if isinstance(formula, U): return any(eval_ltl(formula.b, trace, j) and all(eval_ltl(formula.a, trace, k) for k in range(idx, j)) for j in range(idx, len(trace)))
    if isinstance(formula, S): return any(eval_ltl(formula.b, trace, j) and all(eval_ltl(formula.a, trace, k) for k in range(j+1, idx+1)) for j in range(0, idx+1))
    if isinstance(formula, P): return any(eval_ltl(formula.sub, trace, i) for i in range(0, idx+1))
    if isinstance(formula, H): return all(eval_ltl(formula.sub, trace, i) for i in range(0, idx+1))
    return False
