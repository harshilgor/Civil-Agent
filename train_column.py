"""
Training Loop for Column RL Agent
"""

import matplotlib
matplotlib.use("Agg")

from beams_data import get_num_columns
from column_environment import ColumnEnvironment
from q_learning_agent import QLearningAgent


def run_column_training(num_episodes=50000):
    env   = ColumnEnvironment()
    agent = QLearningAgent(
        num_actions=get_num_columns(),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
    )
    rewards = []

    print("Starting column training...")
    print(f"Episodes: {num_episodes}")
    n_col = get_num_columns()
    print(f"Column sections (W10/W12/W14 only): {n_col}")
    print("=" * 60)

    for episode in range(num_episodes):
        state  = env.reset()
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        agent.update(state, action, reward)
        agent.decay_epsilon()
        rewards.append(reward)

        if (episode + 1) % 5000 == 0:
            avg = sum(rewards[-5000:]) / 5000
            print(f"Episode {episode+1:6d} | "
                  f"Avg reward: {avg:8.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-table: {len(agent.q_table)}")

    print("\nColumn training complete!")
    return agent, rewards


if __name__ == "__main__":
    agent, rewards = run_column_training(num_episodes=50000)
    print(f"Final Q-table size: {len(agent.q_table)}")
