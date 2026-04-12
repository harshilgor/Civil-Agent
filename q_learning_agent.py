"""
Q-Learning Agent for Beam Selection
Learns which beam to pick for any given (span, load) problem
"""

import random


class QLearningAgent:
    """
    Tabular Q-Learning Agent
    
    Q-table: dictionary mapping (state, action) → Q-value
    Example: {((20, 2.0), 5): -31.4} means
             "for span=20, load=2.0, picking beam #5 has value -31.4"
    """
    
    def __init__(self, num_actions, alpha=0.1, gamma=0.9, epsilon=0.3):
        """
        Initialize the agent
        
        Args:
            num_actions: How many beams to choose from (14)
            alpha:       Learning rate (0.1 = learn slowly and steadily)
            gamma:       Discount factor (0.9, less important for 1-step episodes)
            epsilon:     Exploration rate (0.3 = explore 30% of the time)
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = {}
        self.total_episodes = 0
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a (state, action) pair.
        Returns 0.0 if never seen before (optimistic start).
        """
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        """
        Choose which beam to try using epsilon-greedy strategy:
        - With probability epsilon: pick a RANDOM beam (explore)
        - Otherwise: pick the beam with the HIGHEST Q-value (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q = max(q_values)

        # Tie-breaking: collect all actions sharing the max Q-value
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward):
        """
        Update Q-value after taking an action and receiving a reward.
        
        Simplified 1-step formula:
            Q(s,a) ← Q(s,a) + α * (reward - Q(s,a))
        """
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward - current_q)
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self, min_epsilon=0.05, decay_rate=0.9999):
        """
        Slowly reduce exploration over time.
        Agent should explore a lot early, less later.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_best_action(self, state):
        """
        Return the best known action for a state (no exploration).
        Used during evaluation after training is complete.
        """
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
    def get_stats(self):
        """
        Return stats about what the agent has learned.
        """
        return {
            'q_table_size': len(self.q_table),
            'epsilon': round(self.epsilon, 4),
            'total_episodes': self.total_episodes
        }


# Test the agent
if __name__ == "__main__":
    print("Testing Q-Learning Agent")
    print("=" * 60)
    
    agent = QLearningAgent(num_actions=14)
    
    # Test 1: Q-table starts empty, get_q_value returns 0
    print("\nTest 1: Fresh agent Q-values should all be 0")
    state = (20, 2.0)
    for action in range(3):
        q = agent.get_q_value(state, action)
        print(f"  Q({state}, action={action}) = {q}")
    
    # Test 2: Update Q-value and check it changed
    print("\nTest 2: Update with a reward and check Q-value changes")
    agent.update(state, action=5, reward=-31)
    q_after = agent.get_q_value(state, action=5)
    print(f"  After update with reward=-31:")
    print(f"  Q({state}, action=5) = {q_after:.4f}  (expected: ~-3.1)")
    
    # Test 3: Epsilon-greedy (early = lots of random exploration)
    print("\nTest 3: Action selection (epsilon=0.3, expect ~30% random)")
    exploit_count = 0
    explore_count = 0
    for _ in range(1000):
        action = agent.choose_action(state)
        if action == 5:
            exploit_count += 1
        else:
            explore_count += 1
    print(f"  Exploited (picked beam 5): {exploit_count}/1000")
    print(f"  Explored  (picked other):  {explore_count}/1000")
    
    # Test 4: Epsilon decay
    print("\nTest 4: Epsilon decay over 500 episodes")
    print(f"  Start epsilon: {agent.epsilon}")
    for _ in range(500):
        agent.decay_epsilon()
    print(f"  After 500 decays: {agent.epsilon:.4f}  (should be lower)")
    
    # Test 5: get_best_action (no exploration)
    print("\nTest 5: get_best_action always picks highest Q-value")
    best = agent.get_best_action(state)
    print(f"  Best action for {state}: {best}  (should be 5, the only trained beam)")
    
    print("\nAll tests done!")
