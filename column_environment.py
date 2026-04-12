"""
RL Environment for Column Sizing
State: (height, Pu, Mu) -- the design problem
Action: section index -- which W-shape to try
Reward: normalized weight on pass, -1000 on fail
"""

import random
from beams_data import get_num_columns, get_column_by_index
from column_physics import check_column_design


class ColumnEnvironment:

    def __init__(self, height_range=(10, 30),
                 axial_range=(50, 500),
                 moment_range=(0, 200)):
        self.height_range  = height_range
        self.axial_range   = axial_range
        self.moment_range  = moment_range
        self.num_actions   = get_num_columns()
        self.current_height = None
        self.current_pu     = None
        self.current_mu     = None

    def reset(self):
        h  = random.uniform(*self.height_range)
        pu = random.uniform(*self.axial_range)
        mu = random.uniform(*self.moment_range)

        self.current_height = round(h / 5) * 5
        self.current_pu     = round(pu / 50) * 50
        self.current_mu     = round(mu / 50) * 50

        self.current_height = max(self.height_range[0],
                             min(self.height_range[1], self.current_height))
        self.current_pu = max(self.axial_range[0],
                         min(self.axial_range[1], self.current_pu))
        self.current_mu = max(self.moment_range[0],
                         min(self.moment_range[1], self.current_mu))

        return (self.current_height, self.current_pu, self.current_mu)

    def step(self, action):
        col_name, col_props = get_column_by_index(action)
        passes, weight, worst, details = check_column_design(
            self.current_height, self.current_pu,
            self.current_mu, col_props
        )

        if passes:
            min_w, max_w = 10, 850
            normalized = (weight - min_w) / (max_w - min_w)
            reward = -normalized * 100
        else:
            reward = -1000

        state = (self.current_height, self.current_pu, self.current_mu)
        info  = {
            'col_name': col_name,
            'passes':   passes,
            'weight':   weight,
            'worst':    worst,
            'details':  details,
        }
        return state, reward, True, info

    def get_state(self):
        return (self.current_height, self.current_pu, self.current_mu)


if __name__ == "__main__":
    env = ColumnEnvironment()
    print("Column Environment Test")
    print("=" * 60)
    for i in range(5):
        state = env.reset()
        action = random.randint(0, env.num_actions - 1)
        state, reward, done, info = env.step(action)
        print(f"Ep {i+1}: H={state[0]}ft Pu={state[1]}k Mu={state[2]}k-ft  "
              f"col={info['col_name']}  reward={reward:.1f}  "
              f"passes={info['passes']}")
