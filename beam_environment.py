"""
RL Environment for Beam Sizing
State: (span, load) — the design problem
Action: beam index — which beam to try
Reward: normalized -0..-100 on pass (lighter is better), -1000 on fail
"""

import random
from beams_data import get_beam_names, get_num_beams, get_beam_by_index
from beam_physics import check_beam_design


class BeamEnvironment:

    def __init__(self, span_range=(10, 40), load_range=(0.5, 5.0)):
        self.span_range   = span_range
        self.load_range   = load_range
        self.num_actions  = get_num_beams()
        self.current_span = None
        self.current_load = None

    def reset(self):
        span = random.uniform(*self.span_range)
        load = random.uniform(*self.load_range)
        self.current_span = round(span / 10) * 10      # nearest 10 ft
        self.current_load = round(load * 1) / 1        # nearest 1.0 kip/ft
        self.current_span = max(self.span_range[0],
                           min(self.span_range[1], self.current_span))
        self.current_load = max(self.load_range[0],
                           min(self.load_range[1], self.current_load))
        return (self.current_span, self.current_load)

    def step(self, action):
        beam_name, beam_props = get_beam_by_index(action)
        dead = self.current_load * 0.4
        live = self.current_load * 0.6
        passes, weight, worst_ratio, details = check_beam_design(
            self.current_span, dead, live, beam_props
        )
        if passes:
            min_weight = 10
            max_weight = 850
            normalized = (weight - min_weight) / (max_weight - min_weight)
            reward = -normalized * 100
        else:
            reward = -1000
        state  = (self.current_span, self.current_load)
        info   = {
            'beam_name':   beam_name,
            'passes':      passes,
            'weight':      weight,
            'worst_ratio': worst_ratio,
            'details':     details,
        }
        return state, reward, True, info

    def get_state(self):
        return (self.current_span, self.current_load)


if __name__ == "__main__":
    env = BeamEnvironment()
    print("Environment Test")
    print("=" * 60)
    for i in range(5):
        state = env.reset()
        action = random.randint(0, env.num_actions - 1)
        state, reward, done, info = env.step(action)
        print(f"Episode {i+1}: span={state[0]}ft  load={state[1]}k/ft  "
              f"beam={info['beam_name']}  reward={reward:.1f}  "
              f"passes={info['passes']}")