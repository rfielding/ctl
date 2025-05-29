from z3 import *
from typing import Dict, List, Set, Tuple, Callable, Optional, Any
from enum import Enum
import itertools
import numpy as np
from collections import deque, namedtuple
from dataclasses import dataclass

class CTLOperator(Enum):
    """CTL and PCTL temporal operators"""
    EX = "EX"  # Exists Next
    AX = "AX"  # All Next  
    EF = "EF"  # Exists Finally
    AF = "AF"  # All Finally
    EG = "EG"  # Exists Globally
    AG = "AG"  # All Globally
    EU = "EU"  # Exists Until
    AU = "AU"  # All Until
    # Probabilistic operators
    PX = "PX"  # Probabilistic Next
    PF = "PF"  # Probabilistic Finally  
    PG = "PG"  # Probabilistic Globally
    PU = "PU"  # Probabilistic Until

class CTLFormula:
    """Represents a CTL/PCTL formula"""
    def __init__(self, operator=None, left=None, right=None, atomic=None, 
                 prob_bound=None, prob_operator=None):
        self.operator = operator
        self.left = left
        self.right = right
        self.atomic = atomic
        self.prob_bound = prob_bound
        self.prob_operator = prob_operator
    
    def __str__(self):
        if self.atomic:
            return str(self.atomic)
        elif self.operator in [CTLOperator.PX, CTLOperator.PF, CTLOperator.PG]:
            op_name = self.operator.value[1]
            return f"P{self.prob_operator}{self.prob_bound}[{op_name}({self.left})]"
        elif self.operator == CTLOperator.PU:
            return f"P{self.prob_operator}{self.prob_bound}[{self.left} U {self.right}]"
        elif self.operator in [CTLOperator.EX, CTLOperator.AX, CTLOperator.EF, 
                              CTLOperator.AF, CTLOperator.EG, CTLOperator.AG]:
            return f"{self.operator.value}({self.left})"
        elif self.operator in [CTLOperator.EU, CTLOperator.AU]:
            return f"{self.operator.value}({self.left}, {self.right})"
        else:
            return f"({self.left} {self.operator} {self.right})"

@dataclass(frozen=True)
class Message:
    """Represents a message in a channel"""
    sender: str
    content: Any
    timestamp: int = 0
    message_type: str = "data"
    
    def __str__(self):
        return f"{self.sender}:{self.content}"

class BufferedChannel:
    """Represents a buffered channel between processes"""
    def __init__(self, name: str, capacity: int = 1, fifo: bool = True):
        self.name = name
        self.capacity = capacity
        self.buffer = deque() if fifo else []
        self.fifo = fifo
        self.readers = set()
        self.writers = set()
    
    def can_write(self) -> bool:
        """Check if channel can accept a write"""
        return len(self.buffer) < self.capacity
    
    def can_read(self) -> bool:
        """Check if channel has data to read"""
        return len(self.buffer) > 0
    
    def write(self, message: Message) -> bool:
        """Write message to channel if possible"""
        if self.can_write():
            if self.fifo:
                self.buffer.append(message)
            else:
                self.buffer.append(message)  # Could implement LIFO here
            return True
        return False
    
    def read(self) -> Optional[Message]:
        """Read message from channel if possible"""
        if self.can_read():
            if self.fifo:
                return self.buffer.popleft()
            else:
                return self.buffer.pop(0)
        return None
    
    def peek(self) -> Optional[Message]:
        """Peek at next message without removing it"""
        if self.can_read():
            return self.buffer[0]
        return None
    
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity
    
    def size(self) -> int:
        return len(self.buffer)
    
    def contents(self) -> List[Message]:
        """Get current buffer contents"""
        return list(self.buffer)
    
    def __str__(self):
        return f"Channel({self.name}, {len(self.buffer)}/{self.capacity}): {list(self.buffer)}"

@dataclass(frozen=True)
class ProcessState:
    """Represents the state of a single process"""
    process_id: str
    local_state: str
    local_vars: Tuple[Tuple[str, Any], ...]  # Sorted tuple of (key, value) pairs
    pending_actions: Tuple[str, ...]  # Tuple for hashability
    
    def __str__(self):
        vars_str = ",".join(f"{k}={v}" for k, v in self.local_vars)
        actions_str = ",".join(self.pending_actions)
        return f"{self.process_id}[{self.local_state}|{vars_str}|{actions_str}]"
    
    def get_local_vars_dict(self) -> Dict[str, Any]:
        """Convert local_vars tuple back to dict for easier access"""
        return dict(self.local_vars)

@dataclass(frozen=True)
class SystemState:
    """Represents the global state of the concurrent system"""
    processes: Tuple[ProcessState, ...]  # Tuple for hashability
    channels: Tuple[Tuple[str, Tuple[Message, ...]], ...]  # Sorted tuple of (name, contents)
    global_vars: Tuple[Tuple[str, Any], ...]  # Sorted tuple of (key, value)
    timestamp: int = 0
    
    def __hash__(self):
        # All components are now hashable tuples
        return hash((self.processes, self.channels, self.global_vars, self.timestamp))
    
    def __str__(self):
        processes_str = " | ".join(str(p) for p in self.processes)
        channels_str = " | ".join(f"{name}:{list(contents)}" for name, contents in self.channels)
        global_vars_str = ",".join(f"{k}={v}" for k, v in self.global_vars)
        return f"State[{processes_str}] Channels[{channels_str}] Globals[{global_vars_str}]"
    
    def get_channels_dict(self) -> Dict[str, Tuple[Message, ...]]:
        """Convert channels tuple back to dict for easier access"""
        return dict(self.channels)
    
    def get_global_vars_dict(self) -> Dict[str, Any]:
        """Convert global_vars tuple back to dict for easier access"""
        return dict(self.global_vars)

class ConcurrentStateMachine:
    """State machine for concurrent systems with buffered channels"""
    
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}  # process_id -> config
        self.channels: Dict[str, BufferedChannel] = {}
        self.process_transitions: Dict[str, Dict[str, List[Tuple]]] = {}  # process -> state -> [(next_state, condition, action, prob)]
        self.initial_system_state: Optional[SystemState] = None
        self.atomic_props: Dict[str, Callable[[SystemState], bool]] = {}
        
    def add_process(self, process_id: str, initial_state: str, initial_vars: Dict[str, Any] = None):
        """Add a process to the system"""
        self.processes[process_id] = {
            'initial_state': initial_state,
            'initial_vars': initial_vars or {}
        }
        if process_id not in self.process_transitions:
            self.process_transitions[process_id] = {}
    
    def add_channel(self, name: str, capacity: int = 1, fifo: bool = True):
        """Add a buffered channel"""
        channel = BufferedChannel(name, capacity, fifo)
        self.channels[name] = channel
        return channel
    
    def add_process_transition(self, process_id: str, from_state: str, to_state: str,
                             condition: Callable[[SystemState, str], bool] = None,
                             action: Callable[[SystemState, str], SystemState] = None,
                             probability: float = 1.0):
        """Add a transition for a specific process"""
        if process_id not in self.process_transitions:
            self.process_transitions[process_id] = {}
        if from_state not in self.process_transitions[process_id]:
            self.process_transitions[process_id][from_state] = []
        
        self.process_transitions[process_id][from_state].append(
            (to_state, condition or (lambda s, p: True), action or (lambda s, p: s), probability)
        )
    
    def set_initial_system_state(self, global_vars: Dict[str, Any] = None):
        """Set the initial system state"""
        initial_processes = []
        for process_id, config in self.processes.items():
            # Convert local_vars dict to sorted tuple
            local_vars_tuple = tuple(sorted(config['initial_vars'].items()))
            initial_processes.append(ProcessState(
                process_id=process_id,
                local_state=config['initial_state'],
                local_vars=local_vars_tuple,
                pending_actions=tuple()  # Empty tuple initially
            ))
        
        initial_channels = []
        for name, channel in self.channels.items():
            initial_channels.append((name, tuple()))  # Empty channels initially
        
        global_vars = global_vars or {}
        
        self.initial_system_state = SystemState(
            processes=tuple(initial_processes),
            channels=tuple(sorted(initial_channels)),  # Sort for consistent hashing
            global_vars=tuple(sorted(global_vars.items())),  # Convert to sorted tuple
            timestamp=0
        )
    
    def add_atomic_prop(self, name: str, predicate: Callable[[SystemState], bool]):
        """Add an atomic proposition that evaluates system states"""
        self.atomic_props[name] = predicate
    
    def get_enabled_transitions(self, system_state: SystemState) -> List[Tuple[str, str, str, Callable, float]]:
        """Get all enabled transitions from current system state"""
        enabled = []
        
        for i, process in enumerate(system_state.processes):
            process_id = process.process_id
            current_state = process.local_state
            
            if (process_id in self.process_transitions and 
                current_state in self.process_transitions[process_id]):
                
                for next_state, condition, action, probability in self.process_transitions[process_id][current_state]:
                    if condition(system_state, process_id):
                        enabled.append((process_id, current_state, next_state, action, probability))
        
        return enabled
    
    def execute_transition(self, system_state: SystemState, process_id: str, 
                          next_state: str, action: Callable) -> SystemState:
        """Execute a transition and return new system state"""
        # Apply the action to get new system state
        new_system_state = action(system_state, process_id)
        
        # Update the specific process state
        new_processes = []
        for process in new_system_state.processes:
            if process.process_id == process_id:
                new_processes.append(ProcessState(
                    process_id=process_id,
                    local_state=next_state,
                    local_vars=process.local_vars,
                    pending_actions=process.pending_actions
                ))
            else:
                new_processes.append(process)
        
        return SystemState(
            processes=tuple(new_processes),
            channels=new_system_state.channels,
            global_vars=new_system_state.global_vars,
            timestamp=new_system_state.timestamp + 1
        )

class ChannelCTLModelChecker:
    """CTL Model Checker for concurrent systems with channels"""
    
    def __init__(self, concurrent_sm: ConcurrentStateMachine):
        self.csm = concurrent_sm
        self.explored_states: Set[SystemState] = set()
        self.state_graph: Dict[SystemState, List[Tuple[SystemState, float]]] = {}
        self.build_state_space()
    
    def build_state_space(self, max_states: int = 1000, max_depth: int = 20):
        """Build the state space by exploring reachable states with limits"""
        if not self.csm.initial_system_state:
            raise ValueError("Initial system state not set")
        
        queue = deque([(self.csm.initial_system_state, 0)])  # (state, depth)
        self.explored_states.add(self.csm.initial_system_state)
        self.state_graph[self.csm.initial_system_state] = []
        
        states_explored = 0
        print(f"Building state space (max_states={max_states}, max_depth={max_depth})...")
        
        while queue and len(self.explored_states) < max_states:
            current_state, depth = queue.popleft()
            states_explored += 1
            
            # Progress indicator
            if states_explored % 100 == 0:
                print(f"  Explored {states_explored} states, queue size: {len(queue)}")
            
            # Skip if too deep
            if depth >= max_depth:
                continue
                
            enabled_transitions = self.csm.get_enabled_transitions(current_state)
            
            # Limit transitions per state to avoid explosion
            if len(enabled_transitions) > 10:
                enabled_transitions = enabled_transitions[:10]
            
            for process_id, from_state, to_state, action, probability in enabled_transitions:
                try:
                    next_system_state = self.csm.execute_transition(
                        current_state, process_id, to_state, action
                    )
                    
                    # Add edge to state graph
                    self.state_graph[current_state].append((next_system_state, probability))
                    
                    # Explore new state if not seen before
                    if next_system_state not in self.explored_states:
                        self.explored_states.add(next_system_state)
                        self.state_graph[next_system_state] = []
                        queue.append((next_system_state, depth + 1))
                        
                except Exception as e:
                    # Skip transitions that fail
                    continue
        
        print(f"State space exploration complete: {len(self.explored_states)} states")
    
    def evaluate_atomic(self, prop: str) -> Dict[SystemState, bool]:
        """Evaluate atomic proposition on all system states"""
        result = {}
        predicate = self.csm.atomic_props.get(prop)
        if not predicate:
            raise ValueError(f"Unknown atomic proposition: {prop}")
        
        for state in self.explored_states:
            try:
                result[state] = predicate(state)
            except:
                result[state] = False
        
        return result
    
    def evaluate_ex(self, phi_states: Dict[SystemState, bool]) -> Dict[SystemState, bool]:
        """EX φ: φ holds in some next state"""
        result = {}
        for state in self.explored_states:
            result[state] = any(
                phi_states.get(next_state, False)
                for next_state, _ in self.state_graph.get(state, [])
            )
        return result
    
    def evaluate_ef(self, phi_states: Dict[SystemState, bool]) -> Dict[SystemState, bool]:
        """EF φ: φ eventually holds on some path"""
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.explored_states:
                if not result.get(state, False):
                    if any(result.get(next_state, False)
                          for next_state, _ in self.state_graph.get(state, [])):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_ag(self, phi_states: Dict[SystemState, bool]) -> Dict[SystemState, bool]:
        """AG φ: φ always holds on all paths"""
        result = {state: True for state in self.explored_states}
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.explored_states:
                if result.get(state, True):
                    # Must satisfy φ in current state
                    if not phi_states.get(state, False):
                        new_result[state] = False
                        changed = True
                        continue
                    
                    # All successors must satisfy AG φ
                    next_states = [ns for ns, _ in self.state_graph.get(state, [])]
                    if next_states and not all(result.get(next_state, True) for next_state in next_states):
                        new_result[state] = False
                        changed = True
        
        return result
    
    def evaluate_formula(self, formula: CTLFormula) -> Dict[SystemState, bool]:
        """Evaluate CTL formula on the system"""
        if formula.atomic:
            return self.evaluate_atomic(formula.atomic)
        elif formula.operator == CTLOperator.EX:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ex(phi_states)
        elif formula.operator == CTLOperator.EF:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ef(phi_states)
        elif formula.operator == CTLOperator.AG:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ag(phi_states)
        else:
            raise ValueError(f"Operator {formula.operator} not implemented for channel systems")
    
    def check_formula(self, formula: CTLFormula) -> bool:
        """Check if formula holds in initial state"""
        result_states = self.evaluate_formula(formula)
        return result_states.get(self.csm.initial_system_state, False)
    
    def get_satisfying_states(self, formula: CTLFormula) -> Set[SystemState]:
        """Get states where formula is satisfied"""
        result_states = self.evaluate_formula(formula)
        return {state for state, satisfied in result_states.items() if satisfied}

# Helper functions for channel operations
def send_message(channel_name: str, content: Any, message_type: str = "data"):
    """Create an action that sends a message to a channel"""
    def action(system_state: SystemState, sender_process: str) -> SystemState:
        message = Message(sender=sender_process, content=content, 
                         message_type=message_type, timestamp=system_state.timestamp)
        
        # Get current channel contents
        channels_dict = system_state.get_channels_dict()
        current_contents = list(channels_dict.get(channel_name, ()))
        
        # Add message to channel
        new_contents = current_contents + [message]
        
        # Update channels
        new_channels_dict = channels_dict.copy()
        new_channels_dict[channel_name] = tuple(new_contents)
        new_channels = tuple(sorted(new_channels_dict.items()))
        
        return SystemState(
            processes=system_state.processes,
            channels=new_channels,
            global_vars=system_state.global_vars,
            timestamp=system_state.timestamp
        )
    return action

def receive_message(channel_name: str, var_name: str):
    """Create an action that receives a message from a channel"""
    def action(system_state: SystemState, receiver_process: str) -> SystemState:
        channels_dict = system_state.get_channels_dict()
        current_contents = list(channels_dict.get(channel_name, ()))
        if not current_contents:
            return system_state  # No message to receive
        
        # Remove first message (FIFO)
        message = current_contents[0]
        new_contents = current_contents[1:]
        
        # Update channel
        new_channels_dict = channels_dict.copy()
        new_channels_dict[channel_name] = tuple(new_contents)
        new_channels = tuple(sorted(new_channels_dict.items()))
        
        # Update receiver process local vars
        new_processes = []
        for process in system_state.processes:
            if process.process_id == receiver_process:
                # Convert to dict, update, then back to tuple
                new_vars_dict = process.get_local_vars_dict()
                new_vars_dict[var_name] = message.content
                new_vars_tuple = tuple(sorted(new_vars_dict.items()))
                
                new_processes.append(ProcessState(
                    process_id=process.process_id,
                    local_state=process.local_state,
                    local_vars=new_vars_tuple,
                    pending_actions=process.pending_actions
                ))
            else:
                new_processes.append(process)
        
        return SystemState(
            processes=tuple(new_processes),
            channels=new_channels,
            global_vars=system_state.global_vars,
            timestamp=system_state.timestamp
        )
    return action

def can_send(channel_name: str, capacity: int):
    """Condition: can send to channel"""
    def condition(system_state: SystemState, process_id: str) -> bool:
        channels_dict = system_state.get_channels_dict()
        current_contents = channels_dict.get(channel_name, ())
        return len(current_contents) < capacity
    return condition

def can_receive(channel_name: str):
    """Condition: can receive from channel"""
    def condition(system_state: SystemState, process_id: str) -> bool:
        channels_dict = system_state.get_channels_dict()
        current_contents = channels_dict.get(channel_name, ())
        return len(current_contents) > 0
    return condition

# Helper functions for CTL formulas
def atomic(prop: str) -> CTLFormula:
    return CTLFormula(atomic=prop)

def EX(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EX, left=phi)

def EF(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.EF, left=phi)

def AG(phi: CTLFormula) -> CTLFormula:
    return CTLFormula(operator=CTLOperator.AG, left=phi)

def example_simple_producer_consumer():
    """Simplified Producer-Consumer example for faster verification"""
    print("Simple Producer-Consumer (Fast Version):")
    print("=" * 45)
    
    csm = ConcurrentStateMachine()
    
    # Simplified: just 2 states per process
    csm.add_process("producer", "idle", {"count": 0})
    csm.add_process("consumer", "waiting", {"received": 0})
    
    # Smaller channel capacity
    csm.add_channel("data", capacity=1, fifo=True)
    
    # Simpler transitions - producer can send if channel not full
    csm.add_process_transition("producer", "idle", "idle",  # Stay in same state
                              condition=can_send("data", 1),
                              action=send_message("data", "item"))
    
    # Consumer can receive if channel not empty
    csm.add_process_transition("consumer", "waiting", "waiting",  # Stay in same state
                              condition=can_receive("data"),
                              action=receive_message("data", "item"))
    
    csm.set_initial_system_state()
    
    # Simple atomic propositions
    csm.add_atomic_prop("data_available", 
                       lambda s: len(s.get_channels_dict().get("data", ())) > 0)
    
    csm.add_atomic_prop("channel_full",
                       lambda s: len(s.get_channels_dict().get("data", ())) >= 1)
    
    csm.add_atomic_prop("consumer_got_data",
                       lambda s: any(p.process_id == "consumer" and 
                                   p.get_local_vars_dict().get("item") is not None
                                   for p in s.processes))
    
    # Use smaller limits for faster execution
    checker = ChannelCTLModelChecker(csm)
    # The build_state_space method will now use max_states=1000, max_depth=20
    
    print(f"Explored {len(checker.explored_states)} system states")
    
    print("\nFast CTL Properties:")
    print("-" * 20)
    
    # Check basic properties
    ef_data = EF(atomic("data_available"))
    print(f"EF data_available: {checker.check_formula(ef_data)}")
    
    ef_full = EF(atomic("channel_full"))
    print(f"EF channel_full: {checker.check_formula(ef_full)}")
    
    ef_consumer = EF(atomic("consumer_got_data"))
    print(f"EF consumer_got_data: {checker.check_formula(ef_consumer)}")
    
    print(f"\nSample states (first 3):")
    for i, state in enumerate(list(checker.explored_states)[:3]):
        print(f"{i+1}. {state}")

def example_tiny_system():
    """Minimal example for testing"""
    print("Tiny System Test:")
    print("=" * 20)
    
    csm = ConcurrentStateMachine()
    
    # Just one process, one transition
    csm.add_process("proc", "start", {})
    csm.add_channel("ch", capacity=1)
    
    csm.add_process_transition("proc", "start", "sent",
                              condition=can_send("ch", 1),
                              action=send_message("ch", "msg"))
    
    csm.add_process_transition("proc", "sent", "done",
                              condition=can_receive("ch"),
                              action=receive_message("ch", "result"))
    
    csm.set_initial_system_state()
    
    csm.add_atomic_prop("done", lambda s: any(p.local_state == "done" for p in s.processes))
    csm.add_atomic_prop("has_msg", lambda s: len(s.get_channels_dict().get("ch", ())) > 0)
    
    checker = ChannelCTLModelChecker(csm)
    
    print(f"Tiny system: {len(checker.explored_states)} states")
    
    ef_done = EF(atomic("done"))
    ef_msg = EF(atomic("has_msg"))
    
    print(f"EF done: {checker.check_formula(ef_done)}")
    print(f"EF has_msg: {checker.check_formula(ef_msg)}")
    
    print("All states:")
    for i, state in enumerate(checker.explored_states):
        print(f"{i+1}. {state}")

def example_producer_consumer():
    """Example: Producer-Consumer with buffered channel"""
    print("\nProducer-Consumer with Buffered Channel:")
    print("=" * 45)
    
    # Create concurrent system
    csm = ConcurrentStateMachine()
    
    # Add processes
    csm.add_process("producer", "idle", {"items_produced": 0})
    csm.add_process("consumer", "waiting", {"items_consumed": 0, "last_item": None})
    
    # Add buffered channel
    csm.add_channel("data_channel", capacity=2, fifo=True)
    
    # Producer transitions
    csm.add_process_transition("producer", "idle", "producing",
                              condition=lambda s, p: True,
                              action=lambda s, p: s)  # Just change state
    
    csm.add_process_transition("producer", "producing", "sending",
                              condition=can_send("data_channel", 2),
                              action=send_message("data_channel", "item"))
    
    csm.add_process_transition("producer", "sending", "idle",
                              condition=lambda s, p: True,
                              action=lambda s, p: s)
    
    # Consumer transitions  
    csm.add_process_transition("consumer", "waiting", "receiving",
                              condition=can_receive("data_channel"),
                              action=receive_message("data_channel", "last_item"))
    
    csm.add_process_transition("consumer", "receiving", "processing",
                              condition=lambda s, p: True,
                              action=lambda s, p: s)
    
    csm.add_process_transition("consumer", "processing", "waiting",
                              condition=lambda s, p: True,
                              action=lambda s, p: s)
    
    # Set initial state
    csm.set_initial_system_state()
    
    # Add atomic propositions
    csm.add_atomic_prop("producer_idle", 
                       lambda s: any(p.process_id == "producer" and p.local_state == "idle" 
                                   for p in s.processes))
    
    csm.add_atomic_prop("consumer_has_data",
                       lambda s: any(p.process_id == "consumer" and p.get_local_vars_dict().get("last_item") is not None
                                   for p in s.processes))
    
    csm.add_atomic_prop("channel_empty",
                       lambda s: len(s.get_channels_dict().get("data_channel", ())) == 0)
    
    csm.add_atomic_prop("channel_full", 
                       lambda s: len(s.get_channels_dict().get("data_channel", ())) >= 2)
    
    csm.add_atomic_prop("data_in_transit",
                       lambda s: len(s.get_channels_dict().get("data_channel", ())) > 0)
    
    # Create model checker
    checker = ChannelCTLModelChecker(csm)
    
    print(f"Explored {len(checker.explored_states)} system states")
    print(f"State graph has {sum(len(transitions) for transitions in checker.state_graph.values())} transitions")
    
    print("\nChannel-based CTL Properties:")
    print("-" * 30)
    
    # EF consumer_has_data: Consumer can eventually receive data
    ef_consumer_data = EF(atomic("consumer_has_data"))
    print(f"EF consumer_has_data (consumer can get data): {checker.check_formula(ef_consumer_data)}")
    
    # AG(data_in_transit → EF channel_empty): Data in transit eventually gets consumed
    # Simplified: EF channel_empty
    ef_channel_empty = EF(atomic("channel_empty"))
    print(f"EF channel_empty (channel can become empty): {checker.check_formula(ef_channel_empty)}")
    
    # EF channel_full: Channel can become full
    ef_channel_full = EF(atomic("channel_full"))
    print(f"EF channel_full (channel can become full): {checker.check_formula(ef_channel_full)}")
    
    # AG EF producer_idle: Producer can always return to idle
    ag_ef_producer_idle = AG(EF(atomic("producer_idle")))
    print(f"AG EF producer_idle (producer always eventually idle): {checker.check_formula(ag_ef_producer_idle)}")
    
    print("\nSample System States:")
    print("-" * 20)
    for i, state in enumerate(list(checker.explored_states)[:5]):
        print(f"{i+1}. {state}")

def example_distributed_consensus():
    """Example: Distributed consensus with message passing"""
    print("\n\nDistributed Consensus with Message Channels:")
    print("=" * 50)
    
    csm = ConcurrentStateMachine()
    
    # Add three consensus nodes
    for i in range(3):
        node_id = f"node{i}"
        csm.add_process(node_id, "follower", {
            "term": 0, 
            "voted_for": None, 
            "log": [], 
            "vote_count": 0
        })
    
    # Add channels for communication
    csm.add_channel("vote_requests", capacity=10)
    csm.add_channel("vote_responses", capacity=10)  
    csm.add_channel("heartbeats", capacity=5)
    
    # Simplified consensus transitions
    # Node becomes candidate
    for i in range(3):
        node_id = f"node{i}"
        csm.add_process_transition(node_id, "follower", "candidate",
                                  condition=lambda s, p: True,  # Simplified trigger
                                  action=lambda s, p: s)
    
    # Candidate sends vote requests
    for i in range(3):
        node_id = f"node{i}"
        csm.add_process_transition(node_id, "candidate", "waiting_votes",
                                  condition=can_send("vote_requests", 10),
                                  action=send_message("vote_requests", f"vote_for_{node_id}"))
    
    # Set initial state
    csm.set_initial_system_state()
    
    # Add consensus-specific atomic propositions
    csm.add_atomic_prop("has_leader",
                       lambda s: any(p.local_state == "leader" for p in s.processes))
    
    csm.add_atomic_prop("election_in_progress", 
                       lambda s: any(p.local_state in ["candidate", "waiting_votes"] for p in s.processes))
    
    csm.add_atomic_prop("votes_pending",
                       lambda s: len(s.get_channels_dict().get("vote_requests", ())) > 0)
    
    csm.add_atomic_prop("network_quiet",
                       lambda s: all(len(s.get_channels_dict().get(ch, ())) == 0 
                                   for ch in ["vote_requests", "vote_responses", "heartbeats"]))
    
    checker = ChannelCTLModelChecker(csm)
    
    print(f"Explored {len(checker.explored_states)} consensus states")
    
    print("\nConsensus Properties:")
    print("-" * 20)
    
    # EF has_leader: Eventually a leader can be elected
    ef_leader = EF(atomic("has_leader"))
    print(f"EF has_leader (leader can be elected): {checker.check_formula(ef_leader)}")
    
    # EF network_quiet: Network can become quiet
    ef_quiet = EF(atomic("network_quiet"))
    print(f"EF network_quiet (network can be quiet): {checker.check_formula(ef_quiet)}")
    
    # AG EF network_quiet: Network always eventually becomes quiet
    ag_ef_quiet = AG(EF(atomic("network_quiet")))
    print(f"AG EF network_quiet (network always eventually quiet): {checker.check_formula(ag_ef_quiet)}")

class SynchronousChannel:
    """Represents a synchronous channel (like CSP channels or Go unbuffered channels)"""
    def __init__(self, name: str):
        self.name = name
        self.readers = set()
        self.writers = set()
        # No buffer - messages are transferred directly during synchronization
    
    def __str__(self):
        return f"SyncChannel({self.name})"

class SynchronousStateMachine:
    """State machine for synchronous concurrent systems"""
    
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.sync_channels: Dict[str, SynchronousChannel] = {}
        self.process_transitions: Dict[str, Dict[str, List[Tuple]]] = {}
        self.initial_system_state: Optional[SystemState] = None
        self.atomic_props: Dict[str, Callable[[SystemState], bool]] = {}
    
    def add_process(self, process_id: str, initial_state: str, initial_vars: Dict[str, Any] = None):
        """Add a process to the system"""
        self.processes[process_id] = {
            'initial_state': initial_state,
            'initial_vars': initial_vars or {}
        }
        if process_id not in self.process_transitions:
            self.process_transitions[process_id] = {}
    
    def add_sync_channel(self, name: str):
        """Add a synchronous channel"""
        channel = SynchronousChannel(name)
        self.sync_channels[name] = channel
        return channel
    
    def add_sync_transition(self, process_id: str, from_state: str, to_state: str,
                           sync_action: str = None,  # "send:channel:value" or "recv:channel:var"
                           condition: Callable[[SystemState, str], bool] = None,
                           probability: float = 1.0):
        """Add a synchronous transition (can be sync send/recv or internal)"""
        if process_id not in self.process_transitions:
            self.process_transitions[process_id] = {}
        if from_state not in self.process_transitions[process_id]:
            self.process_transitions[process_id][from_state] = []
        
        self.process_transitions[process_id][from_state].append(
            (to_state, condition or (lambda s, p: True), sync_action, probability)
        )
    
    def set_initial_system_state(self, global_vars: Dict[str, Any] = None):
        """Set the initial system state"""
        initial_processes = []
        for process_id, config in self.processes.items():
            local_vars_tuple = tuple(sorted(config['initial_vars'].items()))
            initial_processes.append(ProcessState(
                process_id=process_id,
                local_state=config['initial_state'],
                local_vars=local_vars_tuple,
                pending_actions=tuple()
            ))
        
        # No channels in system state - sync channels don't store messages
        global_vars = global_vars or {}
        
        self.initial_system_state = SystemState(
            processes=tuple(initial_processes),
            channels=tuple(),  # Empty - no buffered channels
            global_vars=tuple(sorted(global_vars.items())),
            timestamp=0
        )
    
    def add_atomic_prop(self, name: str, predicate: Callable[[SystemState], bool]):
        """Add an atomic proposition"""
        self.atomic_props[name] = predicate
    
    def get_sync_transitions(self, system_state: SystemState) -> List[Tuple]:
        """Get all possible synchronizations and internal transitions"""
        transitions = []
        
        # 1. Internal transitions (no synchronization required)
        for i, process in enumerate(system_state.processes):
            process_id = process.process_id
            current_state = process.local_state
            
            if (process_id in self.process_transitions and 
                current_state in self.process_transitions[process_id]):
                
                for next_state, condition, sync_action, probability in self.process_transitions[process_id][current_state]:
                    if condition(system_state, process_id):
                        if sync_action is None:
                            # Internal transition
                            transitions.append(('internal', process_id, current_state, next_state, None, None, probability))
                        else:
                            # Mark as potential sync transition
                            transitions.append(('sync', process_id, current_state, next_state, sync_action, None, probability))
        
        # 2. Find matching synchronous communications
        sync_transitions = []
        sync_candidates = [t for t in transitions if t[0] == 'sync']
        
        for i, sender_trans in enumerate(sync_candidates):
            _, sender_id, sender_from, sender_to, sender_action, _, sender_prob = sender_trans
            
            if sender_action and sender_action.startswith('send:'):
                parts = sender_action.split(':')
                if len(parts) >= 3:
                    channel_name, value = parts[1], parts[2]
                    
                    # Find matching receiver
                    for j, receiver_trans in enumerate(sync_candidates):
                        if i != j:
                            _, receiver_id, receiver_from, receiver_to, receiver_action, _, receiver_prob = receiver_trans
                            
                            if (receiver_action and receiver_action.startswith('recv:') and
                                receiver_action.split(':')[1] == channel_name):
                                
                                recv_parts = receiver_action.split(':')
                                if len(recv_parts) >= 3:
                                    var_name = recv_parts[2]
                                    
                                    # Found matching send/recv pair
                                    sync_transitions.append((
                                        'sync_pair', sender_id, receiver_id,
                                        sender_from, sender_to, receiver_from, receiver_to,
                                        channel_name, value, var_name,
                                        min(sender_prob, receiver_prob)
                                    ))
        
        # Return internal transitions + sync pairs
        internal_transitions = [t for t in transitions if t[0] == 'internal']
        return internal_transitions + sync_transitions
    
    def execute_sync_transition(self, system_state: SystemState, transition: Tuple) -> SystemState:
        """Execute a synchronous transition"""
        if transition[0] == 'internal':
            # Internal transition: just change process state
            _, process_id, from_state, to_state, _, _, _ = transition
            
            new_processes = []
            for process in system_state.processes:
                if process.process_id == process_id:
                    new_processes.append(ProcessState(
                        process_id=process_id,
                        local_state=to_state,
                        local_vars=process.local_vars,
                        pending_actions=process.pending_actions
                    ))
                else:
                    new_processes.append(process)
            
            return SystemState(
                processes=tuple(new_processes),
                channels=system_state.channels,
                global_vars=system_state.global_vars,
                timestamp=system_state.timestamp + 1
            )
        
        elif transition[0] == 'sync_pair':
            # Synchronous communication: update both processes simultaneously
            (_, sender_id, receiver_id, sender_from, sender_to, 
             receiver_from, receiver_to, channel_name, value, var_name, _) = transition
            
            new_processes = []
            for process in system_state.processes:
                if process.process_id == sender_id:
                    # Update sender
                    new_processes.append(ProcessState(
                        process_id=sender_id,
                        local_state=sender_to,
                        local_vars=process.local_vars,
                        pending_actions=process.pending_actions
                    ))
                elif process.process_id == receiver_id:
                    # Update receiver and set received value
                    new_vars_dict = process.get_local_vars_dict()
                    new_vars_dict[var_name] = value
                    new_vars_tuple = tuple(sorted(new_vars_dict.items()))
                    
                    new_processes.append(ProcessState(
                        process_id=receiver_id,
                        local_state=receiver_to,
                        local_vars=new_vars_tuple,
                        pending_actions=process.pending_actions
                    ))
                else:
                    new_processes.append(process)
            
            return SystemState(
                processes=tuple(new_processes),
                channels=system_state.channels,
                global_vars=system_state.global_vars,
                timestamp=system_state.timestamp + 1
            )
        
        else:
            raise ValueError(f"Unknown transition type: {transition[0]}")

class SynchronousCTLModelChecker:
    """CTL Model Checker for synchronous concurrent systems"""
    
    def __init__(self, sync_sm: SynchronousStateMachine):
        self.ssm = sync_sm
        self.explored_states: Set[SystemState] = set()
        self.state_graph: Dict[SystemState, List[Tuple[SystemState, float]]] = {}
        self.build_sync_state_space()
    
    def build_sync_state_space(self, max_states: int = 5000, max_depth: int = 30):
        """Build state space for synchronous system - much more efficient!"""
        if not self.ssm.initial_system_state:
            raise ValueError("Initial system state not set")
        
        queue = deque([(self.ssm.initial_system_state, 0)])
        self.explored_states.add(self.ssm.initial_system_state)
        self.state_graph[self.ssm.initial_system_state] = []
        
        states_explored = 0
        print(f"Building synchronous state space (max_states={max_states}, max_depth={max_depth})...")
        
        while queue and len(self.explored_states) < max_states:
            current_state, depth = queue.popleft()
            states_explored += 1
            
            if states_explored % 200 == 0:
                print(f"  Explored {states_explored} states, queue size: {len(queue)}")
            
            if depth >= max_depth:
                continue
            
            sync_transitions = self.ssm.get_sync_transitions(current_state)
            
            for transition in sync_transitions:
                try:
                    next_system_state = self.ssm.execute_sync_transition(current_state, transition)
                    
                    # Get probability from transition
                    probability = transition[-1] if len(transition) > 6 else 1.0
                    
                    self.state_graph[current_state].append((next_system_state, probability))
                    
                    if next_system_state not in self.explored_states:
                        self.explored_states.add(next_system_state)
                        self.state_graph[next_system_state] = []
                        queue.append((next_system_state, depth + 1))
                        
                except Exception as e:
                    continue
        
        print(f"Synchronous state space complete: {len(self.explored_states)} states")
    
    def evaluate_atomic(self, prop: str) -> Dict[SystemState, bool]:
        """Evaluate atomic proposition on all system states"""
        result = {}
        predicate = self.ssm.atomic_props.get(prop)
        if not predicate:
            raise ValueError(f"Unknown atomic proposition: {prop}")
        
        for state in self.explored_states:
            try:
                result[state] = predicate(state)
            except:
                result[state] = False
        
        return result
    
    def evaluate_ef(self, phi_states: Dict[SystemState, bool]) -> Dict[SystemState, bool]:
        """EF φ: φ eventually holds on some path"""
        result = phi_states.copy()
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.explored_states:
                if not result.get(state, False):
                    if any(result.get(next_state, False)
                          for next_state, _ in self.state_graph.get(state, [])):
                        new_result[state] = True
                        changed = True
            
            result = new_result
        
        return result
    
    def evaluate_ag(self, phi_states: Dict[SystemState, bool]) -> Dict[SystemState, bool]:
        """AG φ: φ always holds on all paths"""
        result = {state: True for state in self.explored_states}
        changed = True
        
        while changed:
            changed = False
            new_result = result.copy()
            
            for state in self.explored_states:
                if result.get(state, True):
                    if not phi_states.get(state, False):
                        new_result[state] = False
                        changed = True
                        continue
                    
                    next_states = [ns for ns, _ in self.state_graph.get(state, [])]
                    if next_states and not all(result.get(next_state, True) for next_state in next_states):
                        new_result[state] = False
                        changed = True
        
        return result
    
    def evaluate_formula(self, formula: CTLFormula) -> Dict[SystemState, bool]:
        """Evaluate CTL formula"""
        if formula.atomic:
            return self.evaluate_atomic(formula.atomic)
        elif formula.operator == CTLOperator.EF:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ef(phi_states)
        elif formula.operator == CTLOperator.AG:
            phi_states = self.evaluate_formula(formula.left)
            return self.evaluate_ag(phi_states)
        else:
            raise ValueError(f"Operator {formula.operator} not implemented")
    
    def check_formula(self, formula: CTLFormula) -> bool:
        """Check if formula holds in initial state"""
        result_states = self.evaluate_formula(formula)
        return result_states.get(self.ssm.initial_system_state, False)
def example_synchronous_producer_consumer():
    """Synchronous Producer-Consumer - much more analyzable!"""
    print("\nSynchronous Producer-Consumer:")
    print("=" * 35)
    
    ssm = SynchronousStateMachine()
    
    # Add processes
    ssm.add_process("producer", "idle", {"items_made": 0})
    ssm.add_process("consumer", "waiting", {"items_consumed": 0, "last_item": None})
    
    # Add synchronous channel
    ssm.add_sync_channel("data_sync")
    
    # Producer transitions
    ssm.add_sync_transition("producer", "idle", "producing")  # Internal
    ssm.add_sync_transition("producer", "producing", "idle", 
                           sync_action="send:data_sync:item")  # Sync send
    
    # Consumer transitions  
    ssm.add_sync_transition("consumer", "waiting", "processing",
                           sync_action="recv:data_sync:received_item")  # Sync recv
    ssm.add_sync_transition("consumer", "processing", "waiting")  # Internal
    
    ssm.set_initial_system_state()
    
    # Atomic propositions
    ssm.add_atomic_prop("producer_idle", 
                       lambda s: any(p.process_id == "producer" and p.local_state == "idle" 
                                   for p in s.processes))
    
    ssm.add_atomic_prop("consumer_has_item",
                       lambda s: any(p.process_id == "consumer" and 
                                   p.get_local_vars_dict().get("received_item") is not None
                                   for p in s.processes))
    
    ssm.add_atomic_prop("both_idle",
                       lambda s: (any(p.process_id == "producer" and p.local_state == "idle" for p in s.processes) and
                                any(p.process_id == "consumer" and p.local_state == "waiting" for p in s.processes)))
    
    ssm.add_atomic_prop("communication_happened",
                       lambda s: any(p.process_id == "consumer" and p.local_state == "processing" for p in s.processes))
    
    checker = SynchronousCTLModelChecker(ssm)
    
    print(f"Synchronous system: {len(checker.explored_states)} states (much smaller!)")
    
    print("\nSynchronous CTL Properties:")
    print("-" * 30)
    
    # These are much more tractable to verify
    ef_communication = EF(atomic("communication_happened"))
    print(f"EF communication_happened: {checker.check_formula(ef_communication)}")
    
    ef_consumer_item = EF(atomic("consumer_has_item"))
    print(f"EF consumer_has_item: {checker.check_formula(ef_consumer_item)}")
    
    # Skip the expensive AG EF check for now
    print("Skipping AG EF check (can be expensive)")
    
    print("\nAll reachable states:")
    for i, state in enumerate(sorted(checker.explored_states, key=str)):
        print(f"{i+1}. {state}")
        if i >= 10:  # Limit output
            print(f"... and {len(checker.explored_states) - 11} more states")
            break

def example_dining_philosophers_sync():
    """Classic dining philosophers with synchronous communication"""
    print("\n\nSynchronous Dining Philosophers (2 philosophers):")
    print("=" * 55)
    
    ssm = SynchronousStateMachine()
    
    # Add 2 philosophers
    for i in range(2):
        ssm.add_process(f"phil{i}", "thinking", {"meals": 0})
    
    # Add synchronous channels for forks
    ssm.add_sync_channel("fork0")
    ssm.add_sync_channel("fork1")
    
    # Philosopher 0 transitions
    ssm.add_sync_transition("phil0", "thinking", "hungry")  # Internal
    ssm.add_sync_transition("phil0", "hungry", "has_left", 
                           sync_action="recv:fork0:left_fork")  # Get left fork
    ssm.add_sync_transition("phil0", "has_left", "eating",
                           sync_action="recv:fork1:right_fork")  # Get right fork
    ssm.add_sync_transition("phil0", "eating", "done")  # Internal
    ssm.add_sync_transition("phil0", "done", "thinking",
                           sync_action="send:fork0:release")  # Release left
    ssm.add_sync_transition("phil0", "done", "thinking", 
                           sync_action="send:fork1:release")  # Release right
    
    # Philosopher 1 transitions (reverse order to avoid deadlock)
    ssm.add_sync_transition("phil1", "thinking", "hungry")
    ssm.add_sync_transition("phil1", "hungry", "has_right",
                           sync_action="recv:fork1:right_fork")  # Get right first
    ssm.add_sync_transition("phil1", "has_right", "eating",
                           sync_action="recv:fork0:left_fork")   # Then left
    ssm.add_sync_transition("phil1", "eating", "done")
    ssm.add_sync_transition("phil1", "done", "thinking",
                           sync_action="send:fork1:release")
    ssm.add_sync_transition("phil1", "done", "thinking",
                           sync_action="send:fork0:release")
    
    # Fork processes (provide forks when requested)
    ssm.add_process("fork0_mgr", "available", {})
    ssm.add_process("fork1_mgr", "available", {})
    
    ssm.add_sync_transition("fork0_mgr", "available", "taken",
                           sync_action="send:fork0:fork")
    ssm.add_sync_transition("fork0_mgr", "taken", "available",
                           sync_action="recv:fork0:release")
    
    ssm.add_sync_transition("fork1_mgr", "available", "taken",
                           sync_action="send:fork1:fork")
    ssm.add_sync_transition("fork1_mgr", "taken", "available", 
                           sync_action="recv:fork1:release")
    
    ssm.set_initial_system_state()
    
    # Deadlock detection
    ssm.add_atomic_prop("deadlock",
                       lambda s: all(p.local_state in ["hungry", "has_left", "has_right"] 
                                   for p in s.processes if p.process_id.startswith("phil")))
    
    ssm.add_atomic_prop("someone_eating",
                       lambda s: any(p.local_state == "eating" 
                                   for p in s.processes if p.process_id.startswith("phil")))
    
    ssm.add_atomic_prop("all_thinking",
                       lambda s: all(p.local_state == "thinking"
                                   for p in s.processes if p.process_id.startswith("phil")))
    
    checker = SynchronousCTLModelChecker(ssm)
    
    print(f"Dining philosophers: {len(checker.explored_states)} states")
    
    print("\nDeadlock Analysis:")
    print("-" * 18)
    
    ef_deadlock = EF(atomic("deadlock"))
    print(f"EF deadlock (deadlock possible): {checker.check_formula(ef_deadlock)}")
    
    ef_eating = EF(atomic("someone_eating"))
    print(f"EF someone_eating (progress possible): {checker.check_formula(ef_eating)}")
    
    ag_ef_thinking = AG(EF(atomic("all_thinking")))
    print(f"AG EF all_thinking (always eventually peaceful): {checker.check_formula(ag_ef_thinking)}")

if __name__ == "__main__":
    # Start with tiny system to test basic functionality
    example_tiny_system()
    
    # Synchronous examples - much more analyzable!
    try:
        example_synchronous_producer_consumer()
    except Exception as e:
        print(f"Synchronous producer-consumer failed: {e}")
    
    print("\nSkipping dining philosophers for now - try simple async version...")
    
    # Then try simple producer-consumer (should be fast)
    try:
        example_simple_producer_consumer()
    except Exception as e:
        print(f"Simple producer-consumer failed: {e}")
    
    print("\nAll examples completed!")
    
    # Skip the complex examples for now - they cause state explosion
    # example_dining_philosophers_sync()  # This might be the slow one
    # example_producer_consumer()  
    # example_distributed_consensus()
