#!/bin/env python3
import unittest
from pobtl_model_checker import *

class TemporalLogicTests(unittest.TestCase):
    def setUp(self):
        # Simple queue model: states 0-3
        self.states = [{"Queue": i} for i in range(4)]
        self.transitions = {}
        
        # Build transitions (can go up/down by 1 or stay)
        for i in range(4):
            current = frozenset({"Queue": i}.items())
            targets = [current]  # Can always stay
            if i > 0:  # Can decrease if > 0
                targets.append(frozenset({"Queue": i-1}.items()))
            if i < 3:  # Can increase if < 3
                targets.append(frozenset({"Queue": i+1}.items()))
            self.transitions[current] = targets
        
        self.model = Model(self.states, self.transitions)
        
        # Basic propositions
        self.q0 = Prop("Queue==0", lambda s: s["Queue"] == 0)
        self.q1 = Prop("Queue==1", lambda s: s["Queue"] == 1)
        self.q2 = Prop("Queue==2", lambda s: s["Queue"] == 2)
        self.q3 = Prop("Queue==3", lambda s: s["Queue"] == 3)
        self.qlt2 = Prop("Queue<2", lambda s: s["Queue"] < 2)

    def test_basic_props(self):
        """Test basic state propositions"""
        state0 = {"Queue": 0}
        self.assertTrue(self.q0.eval(self.model, state0))
        self.assertFalse(self.q1.eval(self.model, state0))

    def test_future_operators(self):
        """Test CTL future operators"""
        state1 = {"Queue": 1}
        
        # EF: Can reach state
        self.assertTrue(EF(self.q0).eval(self.model, state1))  # Can reach 0
        self.assertTrue(EF(self.q3).eval(self.model, state1))  # Can reach 3
        
        # AG: Always globally
        self.assertTrue(AG(Or(self.q0, Or(self.q1, Or(self.q2, self.q3)))).eval(self.model, state1))
        
        # Combined
        self.assertTrue(AG(Implies(self.q3, EF(self.q0))).eval(self.model, state1))

    def test_past_operators(self):
        """Test past temporal operators"""
        state2 = {"Queue": 2}
        
        # Y: Yesterday/Previous
        self.assertTrue(Y(self.q1).eval(self.model, state2))  # Could have been 1
        self.assertTrue(Y(self.q3).eval(self.model, state2))  # Could have been 3
        self.assertFalse(Y(self.q0).eval(self.model, state2))  # Couldn't have been 0
        
        # O: Once in the past
        self.assertTrue(O(self.q1).eval(self.model, state2))
        
        # S: Since
        self.assertTrue(S(Or(self.q1, self.q2), Not(self.q3)).eval(self.model, state2))

    def test_combined_operators(self):
        """Test combinations of past and future operators"""
        state1 = {"Queue": 1}
        
        # Past + Future
        self.assertTrue(AG(Implies(self.q2, Y(self.q1))).eval(self.model, state1))
        self.assertTrue(EF(And(self.q3, O(self.q0))).eval(self.model, state1))
        
        # Complex properties
        always_reachable_zero = AG(EF(self.q0))
        self.assertTrue(always_reachable_zero.eval(self.model, state1))

    def test_edge_cases(self):
        """Test edge cases and potential issues"""
        state0 = {"Queue": 0}
        
        # Self loops
        self.assertTrue(Y(self.q0).eval(self.model, state0))  # Can stay at 0
        
        # Unreachable states
        unreachable = Prop("Queue==5", lambda s: s["Queue"] == 5)
        self.assertFalse(EF(unreachable).eval(self.model, state0))
        self.assertFalse(O(unreachable).eval(self.model, state0))
        
        # Empty paths
        empty_model = Model([], {})
        self.assertFalse(Y(self.q0).eval(empty_model, state0))

    def test_protocol_properties(self):
        """Test typical protocol verification properties"""
        # Protocol model: states with msg and ack flags
        proto_states = [
            {"msg": False, "ack": False},
            {"msg": True, "ack": False},
            {"msg": True, "ack": True}
        ]
        
        proto_transitions = {
            frozenset({"msg": False, "ack": False}.items()): [
                frozenset({"msg": True, "ack": False}.items())
            ],
            frozenset({"msg": True, "ack": False}.items()): [
                frozenset({"msg": True, "ack": True}.items())
            ],
            frozenset({"msg": True, "ack": True}.items()): [
                frozenset({"msg": False, "ack": False}.items())
            ]
        }
        
        proto_model = Model(proto_states, proto_transitions)
        
        msg = Prop("msg", lambda s: s["msg"])
        ack = Prop("ack", lambda s: s["ack"])
        
        # Test: ack implies previous msg
        self.assertTrue(AG(Implies(ack, Y(msg))).eval(proto_model, proto_states[0]))
        
        # Test: msg until ack
        self.assertTrue(EF(And(ack, O(msg))).eval(proto_model, proto_states[0]))

    def test_past_present_future(self):
        """Test properties that involve past, present, and future states"""
        state1 = {"Queue": 1}
        state2 = {"Queue": 2}
        
        # Test: If we're at 2 now, we must have been at 1 or 3 before,
        # and we can reach 0 in the future
        past_present_future = And(
            And(Y(Or(self.q1, self.q3)), self.q2),
            EF(self.q0)
        )
        self.assertTrue(past_present_future.eval(self.model, state2))
        
        # Test: Every state that can reach 3 must have passed through 2
        must_pass_through = AG(
            Implies(
                EF(self.q3),
                Or(self.q2, O(self.q2))  # Either at 2 now or was at 2 in the past
            )
        )
        self.assertTrue(must_pass_through.eval(self.model, state1))
        
        # Test: From any state, if we were at 0 in the past,
        # we must have gone through each number to reach current state
        ordered_path = AG(
            Implies(
                O(self.q0),
                And(
                    Implies(self.q2, O(self.q1)),
                    Implies(self.q3, O(self.q2))
                )
            )
        )
        self.assertTrue(ordered_path.eval(self.model, state1))
        
        # Test: If we reach 3 in the future, and we were at 0 in the past,
        # we must be currently in the middle of the queue (1 or 2)
        middle_of_journey = Implies(
            And(O(self.q0), EF(self.q3)),
            Or(self.q1, self.q2)
        )
        self.assertTrue(middle_of_journey.eval(self.model, state2))
        
        # Test: From any state, if we can reach 0 in the future and were at 3 in the past,
        # we must be either decreasing or staying in place
        decreasing_path = AG(
            Implies(
                And(O(self.q3), EF(self.q0)),
                Or(
                    self.q3,  # Could still be at 3
                    Or(
                        And(self.q2, Y(self.q3)),  # Just moved from 3 to 2
                        Or(
                            And(self.q1, Or(Y(self.q2), O(self.q2))),  # At 1, came through 2
                            And(self.q0, Or(Y(self.q1), O(self.q1)))   # At 0, came through 1
                        )
                    )
                )
            )
        )
        self.assertTrue(decreasing_path.eval(self.model, state2))

if __name__ == '__main__':
    unittest.main()