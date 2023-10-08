class RolloutBuffer:
    def __init__(self):
        # action
        self.actions = []
        # state
        self.node_features = []
        self.graph_features = []
        self.adjacencies = []
        self.candidates = []
        self.masks = []
        # reward
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.terminals = []

    def clear(self):
        del self.actions[:]
        del self.node_features[:]
        del self.graph_features[:]
        del self.adjacencies[:]
        del self.candidates[:]
        del self.masks[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.state_values[:]
        del self.terminals[:]

    def add(self, action, node_feature, graph_feature, adjacency,
            candidate, mask, reward, log_prob, state_value, terminal):
        self.actions.append(action)
        self.node_features.append(node_feature)
        self.graph_features.append(graph_feature)
        self.adjacencies.append(adjacency)
        self.candidates.append(candidate)
        self.masks.append(mask)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.terminals.append(terminal)
