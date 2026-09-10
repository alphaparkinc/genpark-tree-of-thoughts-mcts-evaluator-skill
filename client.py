import math

class ToTMCTS:
    """
    Tree of Thoughts (ToT) with MCTS (Yao et al.).
    Explores reasoning paths as tree nodes with UCB1 exploration-exploitation trade-off.
    """
    def __init__(self, c_param=1.414):
        self.c = c_param
        self.tree = {0: {'state': "root", 'visits': 0, 'value': 0.0, 'children': [], 'parent': None}}
        self.node_counter = 1

    def select(self, node_id=0):
        curr = node_id
        while self.tree[curr]['children']:
            best_child = None
            best_score = -float('inf')
            n_parent = self.tree[curr]['visits']
            for child in self.tree[curr]['children']:
                c_node = self.tree[child]
                if c_node['visits'] == 0:
                    return child
                q = c_node['value'] / c_node['visits']
                u = self.c * math.sqrt(math.log(n_parent) / c_node['visits'])
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child
            curr = best_child
        return curr

    def expand(self, parent_id, thought_candidates):
        for th in thought_candidates:
            cid = self.node_counter
            self.node_counter += 1
            self.tree[cid] = {
                'state': th,
                'visits': 0,
                'value': 0.0,
                'children': [],
                'parent': parent_id
            }
            self.tree[parent_id]['children'].append(cid)
        return self.tree[parent_id]['children'][0] if self.tree[parent_id]['children'] else parent_id

    def backpropagate(self, node_id, reward):
        curr = node_id
        while curr is not None:
            self.tree[curr]['visits'] += 1
            self.tree[curr]['value'] += reward
            curr = self.tree[curr]['parent']
