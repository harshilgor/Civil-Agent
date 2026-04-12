"""
Training Loop
Runs episodes, trains the agent, plots learning curve
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beams_data import get_beam_names, get_num_beams
from beam_environment import BeamEnvironment
from q_learning_agent import QLearningAgent


def run_training(num_episodes=10000):
    env   = BeamEnvironment()
    agent = QLearningAgent(
        num_actions=get_num_beams(),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3
    )
    rewards = []

    print("Starting training...")
    print(f"Episodes: {num_episodes}")
    print(f"Beams in database: {get_num_beams()}")
    print("=" * 60)

    for episode in range(num_episodes):
        state             = env.reset()
        action            = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        agent.update(state, action, reward)
        agent.decay_epsilon()
        rewards.append(reward)

        if (episode + 1) % 1000 == 0:
            avg = sum(rewards[-1000:]) / 1000
            print(f"Episode {episode+1:6d} | "
                  f"Avg reward: {avg:8.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-table: {len(agent.q_table)}")

    print("\nTraining complete!")
    return agent, rewards


def plot_learning_curve(rewards, window=500):
    rolling = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling.append(sum(rewards[start:i+1]) / (i - start + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rolling, color='steelblue', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title(f'Learning curve ({window}-ep rolling avg)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(rewards[:500], color='orange', alpha=0.6, linewidth=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('First 500 episodes (raw)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150)
    plt.close()
    print("Saved to learning_curve.png")


def print_policy(agent):
    beam_names = get_beam_names()
    # Match environment discretization: span to 10 ft, load to 1 kip/ft
    spans = [10, 20, 30, 40]
    loads = [1.0, 2.0, 3.0, 4.0, 5.0]

    print("\n" + "=" * 70)
    print("LEARNED POLICY")
    print("=" * 70)
    print(f"\n{'':12}", end="")
    for load in loads:
        print(f"  {load}k/ft  ", end="")
    print()

    for span in spans:
        print(f"Span {span:2d}ft  ", end="")
        for load in loads:
            action = agent.get_best_action((span, load))
            print(f"  {beam_names[action]:8s}", end="")
        print()


if __name__ == "__main__":
    agent, rewards = run_training(num_episodes=50000)
    print_policy(agent)
    plot_learning_curve(rewards)