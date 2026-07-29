import random

#intialize the game
RANKS = ['J', 'Q', 'K'] # represented by 0,1,2
PASS, BET = 0,1
ACTIONS = ['p','b']
NUM_ACTIONS = len(ACTIONS)

def deal():
    cards = [0,1,2]
    random.shuffle(cards)
    return cards


def is_terminal(history):
    return len(history) >= 2 and history[-2:] in ('pp', 'bp', 'bb')

def payoff(history, cards):  #this function doesn't look at a specific player, it's just whoever's turn it is
    player = len(history) % 2 #because player 1 can only have 0 or 2 histories. represented by 0
    opponent = 1 - player
    player_wins = cards[player] > cards[opponent] 
    if history[-1] == 'p': 
        if history == 'pp': 
            return 1 if player_wins else -1 #check check showdown
        return 1 #bet then opponent folded
    return 2 if player_wins else -2 #called the bet, showdown

def info_set_key(history, cards): #history is what the player sees, e.g Qp having a queen and the game went check
    player = len(history) % 2
    return RANKS[cards[player]] + history
    
#learning state: one per info set
class Node:
    def __init__(self): 
        self.regret_sum = [0.0] * NUM_ACTIONS #regret sum is the amount chips(regrets) of an action over many iterations
        self.strategy_sum = [0.0] * NUM_ACTIONS #how much action a actually got used across training

    #regret matching: the positive regret is the percentage we will take next iteration
    def get_strategy(self, realization_weight): 
        positive = [max(r, 0.0) for r in self.regret_sum] #only act on positive regrets
        total = sum(positive)
        if total > 0:
            strat = [r / total for r in positive]
        else:
            strat = [1.0 / NUM_ACTIONS] * NUM_ACTIONS #just set to equal probability is regret_sum is negative
        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += realization_weight * strat[a] #weighted sum
        return strat

    #exit
    def average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS

#solver 
class KuhnTrainer:
    def __init__(self):  #key is info set key, telling us which part of the tree we are at
        self.nodes = {}

    #lazy look up or create
    def _node(self, key):
        if key not in self.nodes:
            self.nodes[key] = Node()
        return self.nodes[key]
    
    def cfr(self, cards, history, p0, p1): 
        player = len(history) % 2

        if is_terminal(history):
            return payoff(history, cards)

        node = self._node(info_set_key(history, cards)) #fetch the node

        strategy = node.get_strategy(p0 if player == 0 else p1) 

        #intialize util and node_util
        util = [0.0] * NUM_ACTIONS
        node_util = 0.0 

        #recursion
        for a in range(NUM_ACTIONS):
            nxt = history + ACTIONS[a]
            if player == 0:
                util[a] = -self.cfr(cards, nxt, p0 * strategy[a], p1) #update p0
            else:
                util[a] = -self.cfr(cards, nxt, p0, p1 * strategy[a]) #zero sum game, so the value of a branch is negative for the opponent
            node_util += strategy[a] * util[a]

        opp_reach = p1 if player == 0 else p0
        for a in range(NUM_ACTIONS):
            node.regret_sum[a] += opp_reach * (util[a] - node_util) #regret is weighted by how likely opponent could lead to the other branch

        return node_util

    def train(self, iterations):
        total = 0.0
        for _ in range(iterations):
            total += self.cfr(deal(), '', 1.0, 1.0)
        return total / iterations

#validation
def _walk(trainer, cards, history):
    if is_terminal(history):
        return payoff(history, cards)
    node = trainer.nodes.get(info_set_key(history, cards)) #may be unvisited if trained briefly
    avg = node.average_strategy() if node else [1.0 / NUM_ACTIONS] * NUM_ACTIONS
    return sum(avg[a] * -_walk(trainer, cards, history + ACTIONS[a])
                for a in range(NUM_ACTIONS))

def exact_value(trainer):
    #exact EV to P1
    from itertools import permutations
    return sum(_walk(trainer, list(c), '') for c in permutations([0,1,2])) / 6

if __name__ == '__main__':
    random.seed(91)
    trainer = KuhnTrainer()
    value = trainer.train(200000)
 
    print(f'self-play average: {value:+.4f}')
    print(f'exact value of avg strategy: {exact_value(trainer):+.4f}   '
          f'(theory {-1/18:+.4f})\n')
 
    print(f'{"info set":>9}  {"pass":>6}  {"bet":>6}')
    for key in sorted(trainer.nodes, key=lambda k: (len(k), k)):
        avg = trainer.nodes[key].average_strategy()
        print(f'{key:>9}  {avg[PASS]:6.3f}  {avg[BET]:6.3f}')
 
    alpha = trainer.nodes['J'].average_strategy()[BET]
    print(f'\nalpha (P1 bluffs J) = {alpha:.3f}')
    print(f'P1 bets K           = {trainer.nodes["K"].average_strategy()[BET]:.3f}'
          f'   (should be 3*alpha = {3 * alpha:.3f})')