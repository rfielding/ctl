"""
Z3-Native Message Passing Markov Chain Model Runner
Complete example showing how to encode and verify business properties
"""

from z3 import *
import time

def state_var(actor, var_name, time):
    """Create time-indexed state variable"""
    return Int(f"{actor}_{var_name}_{time}")

def channel_var(actor, channel, field, time):
    """Create time-indexed channel variable"""
    return Int(f"{actor}_{channel}_{field}_{time}")

def prob_var(actor, state, time):
    """Create probability variable"""
    return Real(f"{actor}_prob_{state}_{time}")

def create_bakery_system_z3(time_horizon=15):
    """Create complete bakery system in Z3 constraints"""
    constraints = []
    T = time_horizon
    
    print(f"Creating bakery system with {T} time steps...")
    
    for t in range(T-1):
        # Baker (Actor A) variables
        baker_status = state_var("Baker", "status", t)        # 0=idle, 1=producing, 2=delivering
        baker_status_next = state_var("Baker", "status", t+1)
        dough_ready = state_var("Baker", "dough_ready", t)
        dough_ready_next = state_var("Baker", "dough_ready", t+1)
        
        # Shop (Actor B) variables  
        inventory = state_var("Shop", "inventory", t)
        inventory_next = state_var("Shop", "inventory", t+1)
        sales_today = state_var("Shop", "sales", t)
        sales_today_next = state_var("Shop", "sales", t+1)
        
        # Delivery channel variables
        delivery_queue = channel_var("Shop", "delivery", "size", t)
        delivery_queue_next = channel_var("Shop", "delivery", "size", t+1)
        delivery_capacity = 5
        
        # Probability variables
        prob_produce = prob_var("Baker", "produce", t)
        prob_stay_idle = prob_var("Baker", "idle", t)
        prob_sell = prob_var("Shop", "sell", t)
        prob_no_customer = prob_var("Shop", "no_customer", t)
        
        # State predicates
        baker_idle = (baker_status == 0)
        baker_producing = (baker_status == 1) 
        baker_delivering = (baker_status == 2)
        shop_has_inventory = (inventory > 0)
        shop_can_receive = (delivery_queue < delivery_capacity)
        
        # Message passing constraints
        can_deliver = And(baker_delivering, shop_can_receive)
        delivery_happens = And(can_deliver, delivery_queue < delivery_capacity)
        
        # Probability constraints
        constraints.extend([
            # Probabilities sum to 1 and are non-negative
            prob_produce + prob_stay_idle == 1.0,
            prob_produce >= 0, prob_stay_idle >= 0,
            prob_sell + prob_no_customer == 1.0, 
            prob_sell >= 0, prob_no_customer >= 0,
            
            # Set realistic probability values
            prob_produce == 0.8,  # 80% chance baker produces
            prob_sell == 0.6,     # 60% chance of sale when inventory available
        ])
        
        # Baker state transitions
        constraints.extend([
            # Idle → Producing (with probability)
            Implies(And(baker_idle, prob_produce >= 0.8),
                   And(baker_status_next == 1, dough_ready_next == dough_ready + 20)),
            
            # Idle → Stay Idle
            Implies(And(baker_idle, prob_produce < 0.8), 
                   And(baker_status_next == 0, dough_ready_next == dough_ready)),
            
            # Producing → Delivering (when dough ready)
            Implies(And(baker_producing, dough_ready >= 20),
                   And(baker_status_next == 2, dough_ready_next == dough_ready)),
            
            # Producing → Keep Producing (when dough not ready)
            Implies(And(baker_producing, dough_ready < 20),
                   And(baker_status_next == 1, dough_ready_next == dough_ready + 10)),
            
            # Delivering → Idle (after delivery)
            Implies(baker_delivering,
                   And(baker_status_next == 0, dough_ready_next == 0))
        ])
        
        # Shop state transitions
        constraints.extend([
            # Receive delivery
            Implies(delivery_happens,
                   And(inventory_next == inventory + 20,
                       delivery_queue_next == delivery_queue + 1,
                       sales_today_next == sales_today)),
            
            # Make sale (when inventory available and customer arrives)
            Implies(And(shop_has_inventory, prob_sell >= 0.6, Not(delivery_happens)),
                   And(inventory_next == inventory - 1,
                       sales_today_next == sales_today + 1,
                       delivery_queue_next == delivery_queue)),
            
            # No sale (no customer or no inventory)
            Implies(And(Or(Not(shop_has_inventory), prob_sell < 0.6), Not(delivery_happens)),
                   And(inventory_next == inventory,
                       sales_today_next == sales_today,
                       delivery_queue_next == delivery_queue))
        ])
        
        # Domain constraints
        constraints.extend([
            # Variable bounds
            baker_status >= 0, baker_status <= 2,
            dough_ready >= 0, dough_ready <= 100,
            inventory >= 0, inventory <= 200,
            sales_today >= 0, sales_today <= 50,
            delivery_queue >= 0, delivery_queue <= delivery_capacity
        ])
    
    # Initial conditions
    constraints.extend([
        state_var("Baker", "status", 0) == 0,      # Start idle
        state_var("Baker", "dough_ready", 0) == 0, # No dough initially
        state_var("Shop", "inventory", 0) == 0,    # No inventory initially  
        state_var("Shop", "sales", 0) == 0,        # No sales initially
        channel_var("Shop", "delivery", "size", 0) == 0  # Empty delivery queue
    ])
    
    return constraints

def check_eventually_profitable(constraints, time_horizon=15):
    """Check AF(sales > 10): Eventually shop sells more than 10 items"""
    print("\n=== Checking: Eventually Profitable (sales > 10) ===")
    
    solver = Solver()
    solver.add(constraints)
    
    # AF(sales > 10) ≡ ∃t. sales_t > 10
    eventually_profitable = Or([
        state_var("Shop", "sales", t) > 10
        for t in range(time_horizon)
    ])
    
    # Try to find counter-example
    solver.push()
    solver.add(Not(eventually_profitable))
    
    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time
    
    if result == unsat:
        print(f"✅ PROPERTY HOLDS: Shop will eventually be profitable")
        print(f"   Solve time: {solve_time:.3f}s")
        return True
    else:
        print(f"❌ COUNTER-EXAMPLE FOUND:")
        print(f"   Solve time: {solve_time:.3f}s")
        model = solver.model()
        print_counter_example(model, time_horizon)
        return False

def check_inventory_safety(constraints, time_horizon=15):
    """Check AG(inventory <= 100): Inventory never exceeds safe limit"""
    print("\n=== Checking: Inventory Safety (inventory <= 100) ===")
    
    solver = Solver()
    solver.add(constraints)
    
    # AG(inventory <= 100) ≡ ∀t. inventory_t <= 100
    always_safe = And([
        state_var("Shop", "inventory", t) <= 100
        for t in range(time_horizon)
    ])
    
    # Try to find violation
    solver.add(Not(always_safe))
    
    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time
    
    if result == unsat:
        print(f"✅ SAFETY PROPERTY HOLDS: Inventory stays within safe limits")
        print(f"   Solve time: {solve_time:.3f}s")
        return True
    else:
        print(f"❌ SAFETY VIOLATION FOUND:")
        print(f"   Solve time: {solve_time:.3f}s")
        model = solver.model()
        print_safety_violation(model, time_horizon)
        return False

def check_liveness_property(constraints, time_horizon=15):
    """Check EF(baker_status=2): There exists a path where baker delivers"""
    print("\n=== Checking: Liveness (baker eventually delivers) ===")
    
    solver = Solver()
    solver.add(constraints)
    
    # EF(baker delivers) ≡ ∃t. baker_status_t = 2
    eventually_delivers = Or([
        state_var("Baker", "status", t) == 2
        for t in range(time_horizon)
    ])
    
    solver.add(eventually_delivers)
    
    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time
    
    if result == sat:
        print(f"✅ LIVENESS PROPERTY HOLDS: Baker can eventually deliver")
        print(f"   Solve time: {solve_time:.3f}s")
        model = solver.model()
        print_execution_trace(model, time_horizon, focus="delivery")
        return True
    else:
        print(f"❌ LIVENESS PROPERTY FAILS: Baker never delivers")
        print(f"   Solve time: {solve_time:.3f}s")
        return False

def print_counter_example(model, time_horizon):
    """Print execution trace that violates profitability"""
    print("   Execution trace showing why shop never becomes profitable:")
    for t in range(min(10, time_horizon)):  # Show first 10 steps
        baker_status = model.eval(state_var("Baker", "status", t))
        inventory = model.eval(state_var("Shop", "inventory", t)) 
        sales = model.eval(state_var("Shop", "sales", t))
        print(f"   t={t}: Baker={baker_status}, Inventory={inventory}, Sales={sales}")

def print_safety_violation(model, time_horizon):
    """Print the point where safety property is violated"""
    print("   Execution trace showing inventory overflow:")
    for t in range(time_horizon):
        inventory = model.eval(state_var("Shop", "inventory", t))
        if inventory.as_long() > 100:
            print(f"   t={t}: VIOLATION - Inventory={inventory} > 100")
            break
        print(f"   t={t}: Inventory={inventory}")

def print_execution_trace(model, time_horizon, focus=None):
    """Print a sample execution trace"""
    print("   Sample execution trace:")
    for t in range(min(8, time_horizon)):
        baker_status = model.eval(state_var("Baker", "status", t))
        dough = model.eval(state_var("Baker", "dough_ready", t))
        inventory = model.eval(state_var("Shop", "inventory", t))
        sales = model.eval(state_var("Shop", "sales", t))
        
        baker_state_name = ["Idle", "Producing", "Delivering"][baker_status.as_long()]
        
        if focus == "delivery" and baker_status.as_long() == 2:
            print(f"   t={t}: 🚚 Baker={baker_state_name}, Dough={dough}, Inventory={inventory}, Sales={sales}")
        else:
            print(f"   t={t}: Baker={baker_state_name}, Dough={dough}, Inventory={inventory}, Sales={sales}")

def main():
    """Main execution function"""
    print("🍞 Z3-Based Bakery Markov Chain Verification")
    print("=" * 50)
    
    # Create the system
    time_horizon = 12
    constraints = create_bakery_system_z3(time_horizon)
    print(f"Created system with {len(constraints)} constraints")
    
    # Run verification checks
    results = {}
    
    # Check various CTL properties
    results['eventually_profitable'] = check_eventually_profitable(constraints, time_horizon)
    results['inventory_safety'] = check_inventory_safety(constraints, time_horizon)  
    results['liveness'] = check_liveness_property(constraints, time_horizon)
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 VERIFICATION SUMMARY:")
    print(f"  Eventually Profitable: {'✅ PASS' if results['eventually_profitable'] else '❌ FAIL'}")
    print(f"  Inventory Safety:      {'✅ PASS' if results['inventory_safety'] else '❌ FAIL'}")
    print(f"  Delivery Liveness:     {'✅ PASS' if results['liveness'] else '❌ FAIL'}")
    
    # Business insights
    print("\n💼 BUSINESS INSIGHTS:")
    if results['eventually_profitable']:
        print("  • The bakery model will eventually become profitable")
    else:
        print("  • ⚠️  The bakery may never reach profitability - consider adjusting parameters")
    
    if results['inventory_safety']:
        print("  • Inventory levels stay within safe operational limits")
    else:
        print("  • ⚠️  Risk of inventory overflow - may need larger storage")
        
    if results['liveness']:
        print("  • The delivery system is functioning properly")
    else:
        print("  • ⚠️  Delivery system may have deadlock issues")

if __name__ == "__main__":
    main()
