from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Any, Optional
import random
from abc import ABC, abstractmethod

# A Model has a set of Actors
# An Actor is 
# - a collection of states
# - an initial state with an initial dict[str,any] to store variables
# - a collection of input channels
# A State has
# - a precondition, a Callable[[dict[str, any]], bool]
# - a transition List[Pair[probability, State]], to denote which states are next

@dataclass
class Message:
    """Represents a message that can be passed between actors."""
    sender: str
    recipient: str
    content: Any
    timestamp: int = 0

@dataclass
class PendingSend:
    """Represents a pending send operation waiting for a receiver."""
    sender_actor: str
    recipient_actor: str
    value: Any
    variable_name: str  # The variable name the receiver expects

@dataclass
class PendingReceive:
    """Represents a pending receive operation waiting for a sender."""
    receiver_actor: str
    variable_name: str  # Where to store the received value

@dataclass 
class Channel:
    """Represents a communication channel with CSP-style semantics."""
    name: str
    capacity: int = 0  # 0 = synchronous rendezvous
    # For capacity > 0, we store messages in a queue
    message_queue: List[Any] = field(default_factory=list)
    # For synchronous (capacity 0), we track pending operations
    pending_sends: List[PendingSend] = field(default_factory=list)
    pending_receives: List[PendingReceive] = field(default_factory=list)
    
    def is_full(self) -> bool:
        """Check if channel is at capacity."""
        if self.capacity == 0:
            return False  # Synchronous channels are never "full" in the traditional sense
        return len(self.message_queue) >= self.capacity
    
    def can_send_immediate(self) -> bool:
        """Check if a send can complete immediately."""
        if self.capacity == 0:
            return len(self.pending_receives) > 0  # Need a waiting receiver
        else:
            return not self.is_full()
    
    def can_receive_immediate(self) -> bool:
        """Check if a receive can complete immediately."""
        if self.capacity == 0:
            return len(self.pending_sends) > 0  # Need a waiting sender
        else:
            return len(self.message_queue) > 0
    
    def attempt_send(self, sender: str, recipient: str, value: Any, var_name: str) -> bool:
        """Attempt to send. Returns True if successful, False if blocked."""
        if self.capacity == 0:
            # Synchronous: look for matching receiver
            for i, pending_recv in enumerate(self.pending_receives):
                if pending_recv.receiver_actor == recipient:
                    # Found a matching receiver! Complete the rendezvous
                    self.pending_receives.pop(i)
                    return True  # Will be handled by caller to update receiver's variables
            # No receiver ready, add to pending
            self.pending_sends.append(PendingSend(sender, recipient, value, var_name))
            return False  # Blocked
        else:
            # Asynchronous: add to queue if space available
            if not self.is_full():
                # Store the resolved value, not the variable name
                self.message_queue.append((recipient, value, var_name))
                return True
            return False  # Queue full
    
    def attempt_receive(self, receiver: str, var_name: str) -> Tuple[bool, Any]:
        """Attempt to receive. Returns (success, value)."""
        if self.capacity == 0:
            # Synchronous: look for matching sender
            for i, pending_send in enumerate(self.pending_sends):
                if pending_send.recipient_actor == receiver:
                    # Found a matching sender! Complete the rendezvous
                    send_op = self.pending_sends.pop(i)
                    return True, send_op.value
            # No sender ready, add to pending
            self.pending_receives.append(PendingReceive(receiver, var_name))
            return False, None  # Blocked
        else:
            # Asynchronous: take from front of queue (FIFO) if available
            if self.message_queue:
                # Find first message for this receiver
                for i, (recipient, value, _) in enumerate(self.message_queue):
                    if recipient == receiver:
                        # Remove this message and shift others down
                        self.message_queue.pop(i)
                        return True, value
            return False, None
    
    def get_status(self) -> str:
        """Get current channel status for debugging."""
        if self.capacity == 0:
            return f"sync(S:{len(self.pending_sends)},R:{len(self.pending_receives)})"
        else:
            return f"{len(self.message_queue)}/{self.capacity}"

class ChannelOperation:
    """Base class for channel operations in states."""
    pass

@dataclass
class SendOperation(ChannelOperation):
    """Represents a send operation: channel_name ! value"""
    channel_name: str
    recipient: str
    value: Any
    variable_name: str = "msg"  # Default variable name for receiver

@dataclass  
class ReceiveOperation(ChannelOperation):
    """Represents a receive operation: channel_name ? variable_name"""
    channel_name: str
    variable_name: str

class State:
    """Represents a state in an actor's state machine."""
    
    def __init__(self, 
                 name: str,
                 precondition: Callable[[Dict[str, Any]], bool] = None,
                 transitions: List[Tuple[float, 'State']] = None,
                 action: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
                 channel_op: ChannelOperation = None):
        self.name = name
        self.precondition = precondition or (lambda variables: True)
        self.transitions = transitions or []
        self.action = action or (lambda variables: variables)
        self.channel_op = channel_op  # Optional channel operation
    
    def can_enter(self, variables: Dict[str, Any]) -> bool:
        """Check if this state can be entered given current variables."""
        return self.precondition(variables)
    
    def execute_action(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the state's action and return updated variables."""
        return self.action(variables)
    
    def get_next_state(self) -> Optional['State']:
        """Select next state based on transition probabilities."""
        if not self.transitions:
            return None
            
        # Normalize probabilities
        total_prob = sum(prob for prob, _ in self.transitions)
        if total_prob == 0:
            return None
            
        rand_val = random.random() * total_prob
        cumulative = 0
        
        for prob, next_state in self.transitions:
            cumulative += prob
            if rand_val <= cumulative:
                return next_state
        
        # Fallback to last state if rounding errors occur
        return self.transitions[-1][1]

@dataclass
class Actor:
    """Represents an actor in the message passing system."""
    name: str
    states: List[State]
    initial_state: State
    variables: Dict[str, Any] = field(default_factory=dict)
    current_state: Optional[State] = None
    blocked: bool = False  # Track if actor is blocked on channel operation
    
    def __post_init__(self):
        if self.current_state is None:
            self.current_state = self.initial_state
    
    def step(self, channels: Dict[str, Channel], all_actors: Dict[str, 'Actor'] = None) -> Tuple[bool, bool]:
        """Execute one step. Returns (state_changed, operation_completed)."""
        if not self.current_state:
            return False, False

        if all_actors is None:
            all_actors = {}

        operation_completed = False
        
        # Execute state action first (this sets up variables for channel operations)
        if not self.blocked:
            self.variables = self.current_state.execute_action(self.variables)
        
        # Then try channel operations (which may use variables set by the action)
        if self.current_state.channel_op:
            if isinstance(self.current_state.channel_op, SendOperation):
                op = self.current_state.channel_op
                if op.channel_name in channels:
                    channel = channels[op.channel_name]
                    
                    # Resolve the value - if it's a string, look it up in variables
                    actual_value = op.value
                    if isinstance(op.value, str) and op.value in self.variables:
                        actual_value = self.variables[op.value]
                        print(f"    -> DEBUG: Resolved '{op.value}' to {actual_value}")
                    elif isinstance(op.value, str):
                        print(f"    -> DEBUG: Could not resolve '{op.value}', variables: {list(self.variables.keys())}")
                    
                    success = channel.attempt_send(
                        self.name, op.recipient, actual_value, op.variable_name
                    )
                    if success:
                        operation_completed = True
                        self.blocked = False
                        
                        # For synchronous channels, directly set receiver's variable
                        if channel.capacity == 0 and op.recipient in all_actors:
                            all_actors[op.recipient].variables[op.variable_name] = actual_value
                            
                        print(f"    -> {self.name} sent {actual_value} to {op.recipient}")
                        
                        # Extra debug for truck deliveries
                        if self.name == "Truck" and op.channel_name == "bread_to_store":
                            print(f"    -> DEBUG: Truck delivery SUCCESS! Channel capacity: {channel.capacity}")
                    else:
                        self.blocked = True
                        print(f"    -> {self.name} BLOCKED sending to {op.recipient}")
                        
                        # Extra debug for truck delivery failures
                        if self.name == "Truck" and op.channel_name == "bread_to_store":
                            print(f"    -> DEBUG: Truck delivery BLOCKED! Channel status: {channel.get_status()}")
                        
            elif isinstance(self.current_state.channel_op, ReceiveOperation):
                op = self.current_state.channel_op
                if op.channel_name in channels:
                    channel = channels[op.channel_name]
                    success, value = channel.attempt_receive(self.name, op.variable_name)
                    if success:
                        self.variables[op.variable_name] = value
                        operation_completed = True
                        self.blocked = False
                        print(f"    -> {self.name} received {value} into {op.variable_name}")
                        
                        # Extra debug for store receiving payments
                        if self.name == "Store" and op.channel_name == "customer_orders":
                            inventory_count = len(self.variables.get('inventory', []))
                            print(f"    -> DEBUG: Store received payment ${value}, inventory: {inventory_count} items")
                    else:
                        self.blocked = True
                        print(f"    -> {self.name} BLOCKED waiting to receive")
        
        # Attempt state transition only if channel operation completed (or no channel op)
        if not self.blocked:
            if not self.current_state.channel_op or operation_completed:
                next_state = self.current_state.get_next_state()
                if next_state and next_state.can_enter(self.variables):
                    old_state = self.current_state
                    self.current_state = next_state
                    return old_state != next_state, operation_completed
        
        return False, operation_completed

class Model:
    """Represents the entire message passing Markov chain model."""
    
    def __init__(self):
        self.actors: Dict[str, Actor] = {}
        self.channels: Dict[str, Channel] = {}
        self.timestep: int = 0
    
    def add_actor(self, actor: Actor) -> None:
        """Add an actor to the model."""
        self.actors[actor.name] = actor
    
    def add_channel(self, channel: Channel) -> None:
        """Add a communication channel to the model."""
        self.channels[channel.name] = channel
    
    def step(self) -> Dict[str, Tuple[bool, bool]]:
        """Execute one timestep. Returns dict of actor_name -> (state_changed, op_completed)."""
        self.timestep += 1
        results = {}
        
        print(f"\n=== TIMESTEP {self.timestep} ===")
        
        # Show channel status
        channel_status = " | ".join([f"{name}: {channel.get_status()}" for name, channel in self.channels.items()])
        
        for actor_name, actor in self.actors.items():
            old_state = actor.current_state.name if actor.current_state else "None"
            state_changed, op_completed = actor.step(self.channels, self.actors)
            new_state = actor.current_state.name if actor.current_state else "None"
            
            results[actor_name] = (state_changed, op_completed)
            
            # Print actor status
            status = f"{actor_name}: {old_state}"
            if state_changed:
                status += f" -> {new_state}"
            else:
                status += f" (stayed)"
            
            if actor.blocked:
                status += " [BLOCKED]"
            
            if actor.variables:
                status += f" | Variables: {actor.variables}"
            
            print(status)
        
        return results
    
    def run(self, max_steps: int = 20, stop_when_stable: bool = True) -> List[Dict[str, Any]]:
        """Run the model for multiple steps."""
        history = []
        
        print("=== STARTING SIMULATION ===")
        print("Initial state:", {name: actor.current_state.name for name, actor in self.actors.items()})
        
        for step in range(max_steps):
            # Record current state
            current_snapshot = {
                'timestep': self.timestep,
                'actors': {
                    name: {
                        'state': actor.current_state.name if actor.current_state else None,
                        'variables': actor.variables.copy(),
                        'blocked': actor.blocked
                    }
                    for name, actor in self.actors.items()
                }
            }
            history.append(current_snapshot)
            
            # Execute step
            results = self.step()
            
            # Check if system is stable (no state changes and no operations completed)
            state_changes = [changed for changed, _ in results.values()]
            operations = [op_completed for _, op_completed in results.values()]
            
            if stop_when_stable and not any(state_changes) and not any(operations):
                print(f"\n*** System reached stable state at timestep {self.timestep} ***")
                break
        
        print(f"\n=== SIMULATION COMPLETE ===")
        print("Final state:", {name: actor.current_state.name for name, actor in self.actors.items()})
        
        return history


def create_sync_example() -> Model:
    """Create an example with synchronous (capacity 0) communication."""
    
    model = Model()
    
    # Create states for Actor A (sender)
    def inc_counter(variables):
        variables['counter'] = variables.get('counter', 0) + 1
        return variables
    
    a_idle = State("idle")
    a_sending = State("sending", 
                     action=inc_counter,
                     channel_op=SendOperation("sync_channel", "ActorB", 42, "received_value"))
    a_done = State("done")
    
    # Transitions
    a_idle.transitions = [(1.0, a_sending)]
    a_sending.transitions = [(1.0, a_done)]  # Move to done after sending
    
    # Create states for Actor B (receiver)  
    def process_received(variables):
        if 'received_value' in variables:
            variables['processed'] = True
        return variables
    
    b_waiting = State("waiting")
    b_receiving = State("receiving", 
                       channel_op=ReceiveOperation("sync_channel", "received_value"))
    b_processing = State("processing", action=process_received)
    b_done = State("done")
    
    # Transitions - B will sometimes be ready to receive
    b_waiting.transitions = [(0.7, b_waiting), (0.3, b_receiving)]  # 30% chance to be ready
    b_receiving.transitions = [(1.0, b_processing)]  # Always process after receiving
    b_processing.transitions = [(1.0, b_done)]
    
    # Create actors
    actor_a = Actor(
        name="ActorA",
        states=[a_idle, a_sending, a_done],
        initial_state=a_idle,
        variables={'counter': 0}
    )
    
    actor_b = Actor(
        name="ActorB", 
        states=[b_waiting, b_receiving, b_processing, b_done],
        initial_state=b_waiting,
        variables={}
    )
    
    # Add to model
    model.add_actor(actor_a)
    model.add_actor(actor_b)
    model.add_channel(Channel("sync_channel", capacity=0))  # Synchronous!
    
    return model


def create_bakery_business() -> Model:
    """Model a bakery business process with Bakery -> Truck -> Store -> Customer flow."""
    
    model = Model()
    
    # === BAKERY ACTOR ===
    # Bakes different types of bread with prices
    
    def bake_bread(bread_type, price):
        def bake_action(variables):
            variables['breads_baked'] = variables.get('breads_baked', 0) + 1
            variables['production_value'] = variables.get('production_value', 0) + price
            print(f"    -> Bakery baked {bread_type} (${price})")
            return variables
        return bake_action
    
    bakery_idle = State("idle")
    bakery_baking_rye = State("baking_rye", 
                             action=bake_bread("Rye", 12),
                             channel_op=SendOperation("bread_to_truck", "Truck", 
                                                     {"type": "Rye", "price": 12}, "bread"))
    bakery_baking_apple = State("baking_apple",
                               action=bake_bread("Apple", 15), 
                               channel_op=SendOperation("bread_to_truck", "Truck",
                                                       {"type": "Apple", "price": 15}, "bread"))
    bakery_baking_sourdough = State("baking_sourdough",
                                   action=bake_bread("Sourdough", 10),
                                   channel_op=SendOperation("bread_to_truck", "Truck",
                                                           {"type": "Sourdough", "price": 10}, "bread"))
    
    # Bakery transitions - cycles through different breads with some idle time
    bakery_idle.transitions = [(0.4, bakery_baking_rye), (0.3, bakery_baking_apple), 
                              (0.2, bakery_baking_sourdough), (0.1, bakery_idle)]
    bakery_baking_rye.transitions = [(0.7, bakery_idle), (0.3, bakery_baking_apple)]
    bakery_baking_apple.transitions = [(0.7, bakery_idle), (0.3, bakery_baking_sourdough)]
    bakery_baking_sourdough.transitions = [(0.8, bakery_idle), (0.2, bakery_baking_rye)]
    
    # === TRUCK ACTOR ===
    # Receives bread from bakery, delivers to store
    
    def load_bread_action(variables):
        if 'bread' in variables:
            bread = variables['bread']
            variables['cargo'] = variables.get('cargo', [])
            variables['cargo'].append(bread)
            variables['loads_received'] = variables.get('loads_received', 0) + 1
            print(f"    -> Truck loaded {bread['type']} bread")
        return variables
    
    def deliver_bread_action(variables):
        cargo = variables.get('cargo', [])
        print(f"    -> DEBUG: Truck attempting delivery, cargo: {len(cargo)} items")
        if cargo:
            bread = cargo.pop(0)  # Deliver oldest bread first
            variables['deliveries_made'] = variables.get('deliveries_made', 0) + 1
            variables['delivering_bread'] = bread  # Store the bread being delivered
            print(f"    -> Truck delivering {bread['type']} to store")
            return variables
        else:
            print(f"    -> Truck has no cargo to deliver")
            variables['delivering_bread'] = {"type": "None", "price": 0}
            return variables
    
    truck_waiting = State("waiting")
    truck_loading = State("loading",
                         action=load_bread_action,
                         channel_op=ReceiveOperation("bread_to_truck", "bread"))
    truck_traveling = State("traveling")
    truck_delivering = State("delivering",
                            action=deliver_bread_action,
                            channel_op=SendOperation("bread_to_store", "Store", 
                                                    "delivering_bread", "bread_delivery"))
    
    # Truck transitions - more deterministic delivery cycle
    truck_waiting.transitions = [(0.8, truck_loading), (0.2, truck_waiting)]
    truck_loading.transitions = [(0.3, truck_traveling), (0.7, truck_loading)]  # Stay to load more
    truck_traveling.transitions = [(1.0, truck_delivering)]  # Always go to delivering
    truck_delivering.transitions = [(1.0, truck_waiting)]  # Always return to waiting after delivery attempt
    
    # === STORE ACTOR ===
    # Receives bread deliveries, stocks shelves, serves customers
    
    def stock_shelves_action(variables):
        if 'bread_delivery' in variables:
            bread = variables['bread_delivery']
            print(f"    -> DEBUG: Store received bread_delivery: {bread} (type: {type(bread)})")
            
            # Handle case where bread might be a string reference
            if isinstance(bread, str):
                print(f"    -> ERROR: Received string '{bread}' instead of bread object")
                return variables
            elif isinstance(bread, dict) and bread.get('type') != "None":  # Valid delivery
                variables['inventory'] = variables.get('inventory', [])
                variables['inventory'].append(bread)
                variables['items_stocked'] = variables.get('items_stocked', 0) + 1
                print(f"    -> Store stocked {bread['type']} on shelves")
        return variables
    
    def serve_customer_action(variables):
        inventory = variables.get('inventory', [])
        print(f"    -> DEBUG: Store serving, inventory: {len(inventory)} items")
        if 'customer_payment' in variables and inventory:
            payment = variables['customer_payment']
            # Sell the oldest bread
            sold_bread = inventory.pop(0)
            variables['items_sold'] = variables.get('items_sold', 0) + 1
            variables['revenue'] = variables.get('revenue', 0) + sold_bread['price']
            print(f"    -> Store sold {sold_bread['type']} for ${sold_bread['price']} (paid ${payment})")
        elif 'customer_payment' in variables:
            print(f"    -> Store has no bread to sell (customer disappointed)")
        elif inventory:
            print(f"    -> Store has {len(inventory)} items but no customer payment")
        else:
            print(f"    -> Store idle - no inventory, no customers")
        return variables
    
    store_open = State("open")
    store_stocking = State("stocking",
                          action=stock_shelves_action,
                          channel_op=ReceiveOperation("bread_to_store", "bread_delivery"))
    store_serving = State("serving",
                         action=serve_customer_action,
                         channel_op=ReceiveOperation("customer_orders", "customer_payment"))
    store_cleaning = State("cleaning")
    
    # Store transitions - prioritize serving when inventory exists
    store_open.transitions = [(0.3, store_stocking), (0.5, store_serving), (0.2, store_cleaning)]
    store_stocking.transitions = [(0.2, store_open), (0.8, store_serving)]  # Go to serving after stocking
    store_serving.transitions = [(0.3, store_open), (0.7, store_serving)]  # Stay serving longer
    store_cleaning.transitions = [(1.0, store_open)]
    
    # === CUSTOMER ACTOR ===
    # Periodically comes to buy bread
    
    def make_purchase_action(variables):
        variables['purchases_attempted'] = variables.get('purchases_attempted', 0) + 1
        payment = random.choice([10, 12, 15, 20])  # Customer brings different amounts
        variables['money_spent'] = variables.get('money_spent', 0) + payment
        variables['current_payment'] = payment  # Store the payment amount
        print(f"    -> Customer attempting to buy bread with ${payment} (attempt #{variables['purchases_attempted']})")
        return variables
    
    customer_away = State("away")
    customer_shopping = State("shopping",
                             action=make_purchase_action,
                             channel_op=SendOperation("customer_orders", "Store", 
                                                     "current_payment", "customer_payment"))
    customer_leaving = State("leaving")
    
    # Customer transitions - comes more frequently  
    customer_away.transitions = [(0.5, customer_away), (0.5, customer_shopping)]  # 50% chance to shop
    customer_shopping.transitions = [(1.0, customer_leaving)]
    customer_leaving.transitions = [(1.0, customer_away)]
    
    # === CREATE ACTORS ===
    
    bakery = Actor(
        name="Bakery",
        states=[bakery_idle, bakery_baking_rye, bakery_baking_apple, bakery_baking_sourdough],
        initial_state=bakery_idle,
        variables={'breads_baked': 0, 'production_value': 0}
    )
    
    truck = Actor(
        name="Truck",
        states=[truck_waiting, truck_loading, truck_traveling, truck_delivering],
        initial_state=truck_waiting,
        variables={'cargo': [], 'loads_received': 0, 'deliveries_made': 0}
    )
    
    store = Actor(
        name="Store", 
        states=[store_open, store_stocking, store_serving, store_cleaning],
        initial_state=store_open,
        variables={'inventory': [], 'items_stocked': 0, 'items_sold': 0, 'revenue': 0}
    )
    
    customer = Actor(
        name="Customer",
        states=[customer_away, customer_shopping, customer_leaving],
        initial_state=customer_away,
        variables={'purchases_attempted': 0, 'money_spent': 0}
    )
    
    # === ADD TO MODEL ===
    
    model.add_actor(bakery)
    model.add_actor(truck)
    model.add_actor(store)
    model.add_actor(customer)
    
    # Channels with appropriate capacities
    model.add_channel(Channel("bread_to_truck", capacity=2))    # Truck can hold 2 batches
    model.add_channel(Channel("bread_to_store", capacity=3))    # Store has receiving dock space
    model.add_channel(Channel("customer_orders", capacity=3))   # Allow multiple pending orders
    
    return model


def analyze_bakery_metrics(model: Model, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze business metrics from the simulation history."""
    
    final_state = history[-1]['actors'] if history else {}
    
    metrics = {
        'production': {
            'breads_baked': final_state.get('Bakery', {}).get('variables', {}).get('breads_baked', 0),
            'production_value': final_state.get('Bakery', {}).get('variables', {}).get('production_value', 0)
        },
        'logistics': {
            'loads_received': final_state.get('Truck', {}).get('variables', {}).get('loads_received', 0),
            'deliveries_made': final_state.get('Truck', {}).get('variables', {}).get('deliveries_made', 0),
            'cargo_remaining': len(final_state.get('Truck', {}).get('variables', {}).get('cargo', []))
        },
        'retail': {
            'items_stocked': final_state.get('Store', {}).get('variables', {}).get('items_stocked', 0),
            'items_sold': final_state.get('Store', {}).get('variables', {}).get('items_sold', 0),
            'revenue': final_state.get('Store', {}).get('variables', {}).get('revenue', 0),
            'unsold_inventory': len(final_state.get('Store', {}).get('variables', {}).get('inventory', []))
        },
        'customer': {
            'purchases_attempted': final_state.get('Customer', {}).get('variables', {}).get('purchases_attempted', 0),
            'money_spent': final_state.get('Customer', {}).get('variables', {}).get('money_spent', 0)
        }
    }
    
    # Calculated metrics
    waste_rate = 0
    if metrics['production']['breads_baked'] > 0:
        waste_rate = (metrics['production']['breads_baked'] - metrics['retail']['items_sold']) / metrics['production']['breads_baked']
    
    metrics['efficiency'] = {
        'waste_rate': waste_rate,
        'revenue_per_bread': metrics['retail']['revenue'] / max(1, metrics['retail']['items_sold']),
        'customer_satisfaction': metrics['retail']['items_sold'] / max(1, metrics['customer']['purchases_attempted'])
    }
    
    return metrics


def generate_state_machine_diagram(model: Model) -> str:
    """Generate a Mermaid state diagram showing the static Markov chain structure."""
    
    mermaid = ["stateDiagram-v2"]
    
    for actor_name, actor in model.actors.items():
        mermaid.append(f"    state {actor_name} {{")
        
        # Add states for this actor
        for state in actor.states:
            state_id = f"{actor_name}_{state.name}"
            
            # Very simple state names only - no special characters
            label = state.name.replace(":", "_").replace("=", "_").replace("/", "_").replace(" ", "_")
            if state.channel_op:
                if isinstance(state.channel_op, SendOperation):
                    label += "_SEND"
                elif isinstance(state.channel_op, ReceiveOperation):
                    label += "_RECV"
            
            if state != actor.initial_state:
                mermaid.append(f"        {state_id} : {label}")
        
        # Mark initial state
        initial_id = f"{actor_name}_{actor.initial_state.name}"
        initial_label = actor.initial_state.name.replace(":", "_").replace("=", "_").replace("/", "_").replace(" ", "_")
        if actor.initial_state.channel_op:
            if isinstance(actor.initial_state.channel_op, SendOperation):
                initial_label += "_SEND"
            elif isinstance(actor.initial_state.channel_op, ReceiveOperation):
                initial_label += "_RECV"
        
        mermaid.append(f"        {initial_id} : {initial_label}")
        mermaid.append(f"        [*] --> {initial_id}")
        
        # Add transitions with safe labels
        for state in actor.states:
            state_id = f"{actor_name}_{state.name}"
            for prob, next_state in state.transitions:
                next_id = f"{actor_name}_{next_state.name}"
                # Only show probability if not 1.0, and keep it simple
                if prob == 1.0:
                    mermaid.append(f"        {state_id} --> {next_id}")
                else:
                    # Format probability as simple decimal
                    prob_str = f"{prob:.1f}".replace("0.", ".")
                    mermaid.append(f"        {state_id} --> {next_id} : {prob_str}")
        
        mermaid.append("    }")
        mermaid.append("")
    
    return "\n".join(mermaid)


def generate_interaction_diagram(model: Model, history: List[Dict[str, Any]], max_steps: int = 15) -> str:
    """Generate a Mermaid sequence diagram showing message passing during a run."""
    
    mermaid = ["sequenceDiagram"]
    
    # Add participants
    actor_names = list(model.actors.keys())
    for name in actor_names:
        mermaid.append(f"    participant {name}")
    
    mermaid.append("")
    
    # Look for evidence of successful message transfers in the history
    messages = []
    
    for i, snapshot in enumerate(history):
        timestep = snapshot['timestep']
        actors = snapshot['actors']
        
        # Look for variable changes that indicate message receipt
        for actor_name, actor_data in actors.items():
            variables = actor_data['variables']
            
            # Check for Truck receiving bread (loads_received increased)
            if actor_name == "Truck" and variables.get('loads_received', 0) > 0:
                # This timestep shows a bread delivery to truck
                if i == 0 or variables.get('loads_received', 0) > history[i-1]['actors'].get('Truck', {}).get('variables', {}).get('loads_received', 0):
                    messages.append({
                        'step': timestep,
                        'from': 'Bakery',
                        'to': 'Truck',
                        'message': 'bread_to_truck bread'
                    })
            
            # Check for Store receiving deliveries (items_stocked increased)
            if actor_name == "Store" and variables.get('items_stocked', 0) > 0:
                if i == 0 or variables.get('items_stocked', 0) > history[i-1]['actors'].get('Store', {}).get('variables', {}).get('items_stocked', 0):
                    messages.append({
                        'step': timestep,
                        'from': 'Truck',
                        'to': 'Store',
                        'message': 'bread_to_store delivery'
                    })
            
            # Check for Store receiving payments (customer_payment variable exists)
            if actor_name == "Store" and 'customer_payment' in variables:
                # Check if this is a new payment
                prev_payment = None
                if i > 0:
                    prev_payment = history[i-1]['actors'].get('Store', {}).get('variables', {}).get('customer_payment')
                
                if prev_payment != variables.get('customer_payment'):
                    messages.append({
                        'step': timestep,
                        'from': 'Customer',
                        'to': 'Store',
                        'message': f"customer_orders ${variables.get('customer_payment', 0)}"
                    })
    
    # Remove duplicates and sort by step
    seen = set()
    unique_messages = []
    for msg in messages:
        key = (msg['step'], msg['from'], msg['to'], msg['message'])
        if key not in seen:
            seen.add(key)
            unique_messages.append(msg)
    
    unique_messages.sort(key=lambda x: x['step'])
    
    # Generate sequence diagram
    for msg in unique_messages:
        mermaid.append(f"    {msg['from']}->>{msg['to']}: Step {msg['step']}: {msg['message']}")
    
    if unique_messages:
        mermaid.append(f"    Note right of {actor_names[0]}: {len(unique_messages)} messages detected")
    else:
        mermaid.append(f"    Note right of {actor_names[0]}: No message transfers detected")
    
    return "\n".join(mermaid)


def generate_metrics_timeline(history: List[Dict[str, Any]]) -> str:
    """Generate a chart showing key business metrics over time."""
    
    mermaid = ["xychart-beta"]
    mermaid.append('    title "Bakery Business Metrics Over Time"')
    mermaid.append('    x-axis "Timestep" 1 --> ' + str(len(history)))
    mermaid.append('    y-axis "Count" 0 --> 20')
    
    # Extract metrics over time
    breads_baked = []
    items_sold = []
    revenue = []
    
    for snapshot in history:
        bakery_vars = snapshot['actors'].get('Bakery', {}).get('variables', {})
        store_vars = snapshot['actors'].get('Store', {}).get('variables', {})
        
        breads_baked.append(bakery_vars.get('breads_baked', 0))
        items_sold.append(store_vars.get('items_sold', 0))
        revenue.append(store_vars.get('revenue', 0) // 10)  # Scale down for chart
    
    mermaid.append('    line "Breads Baked" [' + ','.join(map(str, breads_baked)) + ']')
    mermaid.append('    line "Items Sold" [' + ','.join(map(str, items_sold)) + ']')
    mermaid.append('    line "Revenue ($10s)" [' + ','.join(map(str, revenue)) + ']')
    
    return "\n".join(mermaid)


def print_diagrams(model: Model, history: List[Dict[str, Any]]):
    """Print all diagrams in a format suitable for documentation."""
    
    print("\n" + "="*60)
    print("# STATE MACHINE DIAGRAM")
    print("="*60)
    print("\n```mermaid")
    print(generate_state_machine_diagram(model))
    print("```")
    
    print("\n" + "="*60) 
    print("# INTERACTION DIAGRAM")
    print("="*60)
    print("\n```mermaid")
    print(generate_interaction_diagram(model, history))
    print("```")
    
    print("\n" + "="*60)
    print("# METRICS TIMELINE")
    print("="*60)
    print("\n```mermaid")
    print(generate_metrics_timeline(history))
    print("```")


if __name__ == "__main__":
    print("# Message Passing Markov Chain Business Process Simulation")
    print("\nThis document contains an executable specification of a bakery business process using Message Passing Markov Chains.")
    print("\n## Synchronous Channel Example")
    print("\n```")
    sync_model = create_sync_example()
    sync_model.run(max_steps=15)
    print("```")
    
    print("\n## Bakery Business Process Simulation")
    print("\n```")
    bakery_model = create_bakery_business()
    history = bakery_model.run(max_steps=25, stop_when_stable=False)
    print("```")
    
    print("\n## Business Metrics Analysis")
    metrics = analyze_bakery_metrics(bakery_model, history)
    
    print(f"\n- **Production**: {metrics['production']['breads_baked']} breads baked, ${metrics['production']['production_value']} value")
    print(f"- **Logistics**: {metrics['logistics']['loads_received']} loads received, {metrics['logistics']['deliveries_made']} deliveries made")
    print(f"- **Retail**: {metrics['retail']['items_stocked']} stocked, {metrics['retail']['items_sold']} sold, ${metrics['retail']['revenue']} revenue")
    print(f"- **Customer**: {metrics['customer']['purchases_attempted']} attempts, ${metrics['customer']['money_spent']} spent")
    print(f"- **Efficiency**: {metrics['efficiency']['waste_rate']:.2%} waste rate, ${metrics['efficiency']['revenue_per_bread']:.2f} revenue/bread")
    print(f"- **Satisfaction**: {metrics['efficiency']['customer_satisfaction']:.2%} customer satisfaction")
    
    print(f"\n**Inventory Status:**")
    print(f"- Unsold inventory: {metrics['retail']['unsold_inventory']} items")
    print(f"- Truck cargo remaining: {metrics['logistics']['cargo_remaining']} items")
    
    print(f"\n## Temporal Logic Questions")
    print(f"\nWith this executable model, we can now ask CTL-style temporal logic questions:")
    print(f"\n- **Safety**: Did we ever have zero inventory? (Check if Store.inventory was ever empty)")
    print(f"- **Liveness**: What's the maximum revenue we can achieve? (Analyze revenue growth patterns)")
    print(f"- **Reachability**: Is waste inevitable? (Check if production always exceeds sales)")
    print(f"- **Optimization**: Can customers always find bread? (Check inventory vs. customer arrival patterns)")
    print(f"- **Performance**: What's the optimal production rate? (Minimize waste while maximizing revenue)")
    
    # Generate diagrams
    print_diagrams(bakery_model, history)
    
    print(f"\n---")
    print(f"\n**Generated by Message Passing Markov Chain Framework**")
    print(f"\n*This specification can be version controlled, tested, and formally verified.*")