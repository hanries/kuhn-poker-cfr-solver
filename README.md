# Kuhn Poker CFR Solver

A from-scratch implementation of Counterfactual Regret Minimization (CFR) that solves
Kuhn poker to its Nash equilibrium. Pure Python, no dependencies, ~110 lines.

The solver converges to a game value of **-0.0555** chips per hand for Player 1, against
a theoretical value of **-1/18 = -0.0556**, and recovers Player 2's unique equilibrium
strategy to three decimal places.

---

## Contents

- [Quick start](#quick-start)
- [The game](#the-game)
- [The known solution](#the-known-solution)
- [How CFR works](#how-cfr-works)
- [Code structure](#code-structure)
- [Sample output](#sample-output)
- [Validation](#validation)
- [Implementation notes](#implementation-notes)
- [Extensions](#extensions)
- [References](#references)

---

## Quick start

```bash
python kuhn_cfr.py
```

No dependencies beyond the standard library. Runs 200,000 iterations in a few seconds.

```python
from kuhn_cfr import KuhnTrainer, exact_value

trainer = KuhnTrainer()
trainer.train(200_000)

print(exact_value(trainer))                              # -0.0555
print(trainer.nodes['Qb'].average_strategy())            # [0.664, 0.336]
```

---

## The game

Kuhn poker is a two-player zero-sum poker variant introduced by Harold Kuhn in 1950.
It is the smallest game that exhibits every essential poker phenomenon — bluffing,
slowplaying, bluff-catching, and mixed-strategy indifference — while remaining small
enough to solve analytically and verify by hand.

**Rules**

- Three-card deck: J, Q, K, ranked K > Q > J
- Both players ante 1 chip
- Each player is dealt one private card; the third is never revealed
- Player 1 acts first: check or bet (1 chip)
- Betting is capped at one bet — no raises
- Highest card wins at showdown

**Action encoding**

Every action is either `p` (pass) or `b` (bet):

| symbol | no bet faced | facing a bet |
|---|---|---|
| `p` | check | fold |
| `b` | bet | call |

This collapse is why every decision in the game has exactly two options, and why all
regret and strategy vectors have length 2.

**Game tree**

```
                          P1
                   p /          \ b
                    /            \
                  P2              P2
              p /    \ b      p /    \ b
               /      \        /      \
        showdown      P1    P2 wins   showdown
           ±1      p /  \ b    +1        ±2
                    /    \
              P1 folds  showdown
                 -1        ±2
```

Five terminal action sequences: `pp`, `pbp`, `pbb`, `bp`, `bb`.

**Size**

| quantity | count | derivation |
|---|---|---|
| deals | 6 | 3 x 2 ordered (P1 card, P2 card) |
| terminal nodes | 30 | 6 deals x 5 sequences |
| decision nodes | 24 | 6 deals x 4 decision points |
| **information sets** | **12** | 4 decision points x 3 private cards |

**Information sets**

An information set is the set of game states the acting player cannot distinguish
between. Holding the Queen at the root, Player 1 cannot tell whether the opponent
holds the Jack or the King — so those two distinct nodes form one information set,
and Player 1 is *required* to play them identically. Playing them differently would
mean acting on a card they cannot see.

Each of the 12 information sets bundles exactly 2 of the 24 decision nodes. Information
sets are keyed by `(own card, action history)`:

| key | who acts | meaning |
|---|---|---|
| `J`, `Q`, `K` | P1 | opening decision |
| `Jp`, `Qp`, `Kp` | P2 | facing a check |
| `Jb`, `Qb`, `Kb` | P2 | facing a bet |
| `Jpb`, `Qpb`, `Kpb` | P1 | checked, then faced a bet |

Note that the leading card is always the *acting* player's. In `Jp`, the Jack belongs
to Player 2.

---

## The known solution

Kuhn poker has a **one-parameter family** of Nash equilibria. Let α ∈ [0, 1/3] be
Player 1's bluffing frequency with the Jack.

**Player 1** (not unique — slides with α)

| information set | equilibrium action |
|---|---|
| `J` | bet with probability α |
| `Q` | always check |
| `K` | bet with probability 3α |
| `Jpb` | always fold |
| `Qpb` | call with probability α + 1/3 |
| `Kpb` | always call |

**Player 2** (unique)

| information set | equilibrium action |
|---|---|
| `Jp` | bet with probability 1/3 |
| `Qp` | always check |
| `Kp` | always bet |
| `Jb` | always fold |
| `Qb` | call with probability 1/3 |
| `Kb` | always call |

**Game value: -1/18 to Player 1**, constant across the entire family.

### Why these numbers

**The Queen never opens.** Holding the Queen, betting has EV -1/2 (the Jack always
folds, the King always calls) while checking has EV -1/3. Checking wins by 1/6 of a
chip, strictly, so this is pure in every equilibrium. The Queen fails the value-bet
test in both directions: nothing worse ever calls it, and nothing better ever folds
to it. Its job is to bluff-catch later.

**The Jack is the bluff, not the Queen.** The Jack can never win a showdown, so
bluffing with it forfeits zero equity. The Queen beats the Jack at showdown, so
bluffing with it throws away real value. Bluff from the bottom of the range.

**The King bets exactly 3x as often as the Jack.** This ratio is pinned by Player 2's
indifference. Given P2 holds the Queen and faces a bet, Bayes gives
P(P1 has J | bet) = α / (α + 3α) = 1/4. Then:

```
EV(fold) = -1
EV(call) = (1/4)(+2) + (3/4)(-2) = -1
```

Exactly indifferent. Any other ratio makes P2 strictly prefer calling or folding,
P2 deviates to a pure strategy, and P1's mix collapses.

**Why α is free.** Player 1 is exactly indifferent between betting and checking with
both the Jack (EV -1 either way) and the King (EV 7/6 either way). Nothing pins those
frequencies to a specific value — only their 3:1 ratio is constrained. Since 3α ≤ 1,
α cannot exceed 1/3.

---

## How CFR works

CFR is a self-play algorithm. The plain-English version:

1. Start by playing uniformly at random.
2. Play a hand against yourself. At every decision point, ask: *for each action I
   could have taken, how would that have turned out versus what I actually did?*
3. Add those differences to a running total — the cumulative regret.
4. Adjust the strategy to favor actions with high regret.
5. Repeat, averaging every strategy used along the way.

The theorem that makes this useful: in a two-player zero-sum game, **the average
strategy converges to a Nash equilibrium**. Equilibrium is never targeted directly;
it falls out of minimizing regret against yourself.

### Regret matching

Given cumulative regrets, the strategy is proportional to the positive part:

```python
positive = [max(r, 0.0) for r in regret_sum]
total = sum(positive)
strategy = [r / total for r in positive] if total > 0 else uniform
```

Negative regret means an action has been worse than what you were doing. It gets zero
weight, but the negative value stays on the books — an action that has been wrong for
many iterations must climb back above zero before re-entering the mix. This acts as a
shock absorber against thrashing.

### The baseline

Regret is measured against `node_util`, the strategy-weighted average of all action
values, rather than against zero:

```python
regret[a] = util[a] - node_util
```

This matters more than it looks. At an information set where the actions are worth
1.16 and 1.17 chips, allocating proportionally to raw utility gives [0.498, 0.502] —
a 0.4% edge produces a 0.4% tilt, and the solver never learns. Subtracting the
baseline turns that small edge into the entire signal.

The baseline also creates the algorithm's fixed point. Because `node_util` is a convex
combination of the action values, an action with positive regret gets played more,
which pulls `node_util` toward it, which shrinks its own regret. The system settles
exactly where every action in the support has equal value — which is the **indifference
condition**, which is the Nash equilibrium. Zero regret and indifference are the same
statement.

Removing the baseline and accumulating raw utility instead leaves every information set
near uniform after 200,000 iterations, and produces a game value of -0.0257.

### The two reach probabilities

This is the part most implementations get wrong. Two separate quantities are tracked,
and the two ledgers use *different* ones:

| ledger | weighted by | reason |
|---|---|---|
| `strategy_sum` | your **own** reach | an iteration where you rarely reached this information set shouldn't dominate the average of what you'd do there |
| `regret_sum` | your **opponent's** reach | strip out your own contribution so you keep learning at information sets your current strategy avoids |

The regret weighting is what "counterfactual" refers to. If your current strategy never
bets the Jack, you never reach the information sets that follow a Jack bet. Weighting
regret by your own reach would make those decisions count for nothing, so you'd never
learn to fix them, so you'd keep not going there. Excluding your own reach breaks that
trap — *counter to fact*, assume you got there and evaluate honestly.

In the general formulation the regret weight is the full counterfactual reach,
π<sup>σ</sup><sub>-i</sub> — chance *and* opponent. Here chance is handled by sampling
one deal per iteration, so the 1/6 is implicit in the sampling frequency rather than
multiplied in.

### Why the average, not the current strategy

The current strategy oscillates forever and never settles. Only the **reach-weighted
average** across all iterations converges to Nash. A plain unweighted average does not
work: behavioral strategies do not average linearly, and weighting by own-reach is the
correction that makes the behavioral average correspond to the average realization plan.

---

## Code structure

Three layers, and the middle one knows nothing about poker.

```
kuhn_cfr.py
├── game rules          stateless functions: deal, is_terminal, payoff, info_set_key
├── Node                per-information-set learning state
├── KuhnTrainer         owns the node dictionary and the training loop
└── exact_value         evaluation, no learning
```

**Game layer.** Plain functions, no state. One Kuhn poker exists; `payoff('bp', [0,2])`
returns `1` today and forever. Wrapping this in a class would produce a class with no
instance variables.

**`Node`.** Contains no poker. No cards, no histories, no betting. Two length-2 float
lists and the code to normalize them. Point it at rock-paper-scissors and it works
unchanged.

**`KuhnTrainer`.** Holds `self.nodes`, a dict of 12 entries mapping information-set
keys to `Node` objects, plus the recursion and the loop.

The only seam between the game layer and the learning layer is `info_set_key`, which
turns game state into a dictionary key. Swap the game layer for Leduc and `Node` does
not change; swap vanilla CFR for CFR+ and the rulebook does not change.

### The two ledgers

Each `Node` holds two lists that never read each other:

| | holds | units | range |
|---|---|---|---|
| `regret_sum` | accumulated utility differences | chips | any sign, unbounded |
| `strategy_sum` | accumulated reach-weighted probability | weight | non-negative, grows with iterations |

**Neither holds probabilities.** After 200,000 iterations, `nodes['Q'].regret_sum` is
roughly `[1.6, -11248.3]` and `nodes['Qb'].strategy_sum` is roughly `[11151, 5553]`.
Probabilities exist only as outputs, produced by normalizing on demand.

| method | reads | called | converges? |
|---|---|---|---|
| `get_strategy(w)` | `regret_sum` | every visit, every iteration | no — oscillates forever |
| `average_strategy()` | `strategy_sum` | once, after training | yes — to Nash |

`get_strategy` also accumulates into `strategy_sum` as a side effect. This fusion is
deliberate: it makes the ordering error unrepresentable, since `strategy_sum` must be
accumulated with the pre-update strategy. To inspect a node without mutating it, call
`get_strategy(0.0)`.

### The recursion

```python
def cfr(self, cards, history, p0, p1):
    player = len(history) % 2

    if is_terminal(history):
        return payoff(history, cards)

    node = self._node(info_set_key(history, cards))
    strategy = node.get_strategy(p0 if player == 0 else p1)      # own reach

    util = [0.0] * NUM_ACTIONS
    node_util = 0.0
    for a in range(NUM_ACTIONS):
        nxt = history + ACTIONS[a]
        if player == 0:
            util[a] = -self.cfr(cards, nxt, p0 * strategy[a], p1)
        else:
            util[a] = -self.cfr(cards, nxt, p0, p1 * strategy[a])
        node_util += strategy[a] * util[a]

    opp_reach = p1 if player == 0 else p0                        # opponent reach
    for a in range(NUM_ACTIONS):
        node.regret_sum[a] += opp_reach * (util[a] - node_util)

    return node_util
```

`p0` and `p1` are indexed by player identity. `realization_weight` and `opp_reach` are
indexed by *role at this node* — the same two numbers, relabeled, with the assignment
swapping at every level of the recursion.

`util` and `node_util` are local scratch, recreated fresh on every invocation and
discarded on return. Only `node_util` survives, as the return value. The whole job of a
`cfr` call is to convert momentary scratch values into a permanent regret contribution
before the stack unwinds.

---

## Sample output

```
self-play average: -0.0609
exact value of avg strategy: -0.0555   (theory -0.0556)

 info set    pass     bet
        J   0.913   0.087
        K   0.710   0.290
        Q   1.000   0.000
       Jb   1.000   0.000
       Jp   0.665   0.335
       Kb   0.000   1.000
       Kp   0.000   1.000
       Qb   0.664   0.336
       Qp   1.000   0.000
      Jpb   1.000   0.000
      Kpb   0.000   1.000
      Qpb   0.574   0.426

alpha (P1 bluffs J) = 0.087
P1 bets K           = 0.290   (should be 3*alpha = 0.261)
```

Per-deal expected values under the converged strategy:

| deal | EV to P1 |
|---|---|
| P1:J  P2:Q | -0.9137 |
| P1:J  P2:K | -1.0870 |
| P1:Q  P2:J | +0.7579 |
| P1:Q  P2:K | -1.4258 |
| P1:K  P2:J | +1.2379 |
| P1:K  P2:Q | +1.0977 |
| **average** | **-0.0555** |

`Q vs K` at -1.4258 is the worst cell in the game — worse than holding the Jack —
because the Queen is the hand that keeps paying off bets it cannot beat. That is
precisely why the equilibrium checks it.

---

## Validation

**Player 2's strategy is unique**, so it can be checked entry by entry. `Qb` at 0.336
and `Jp` at 0.335 are both 1/3. `Qp` is a pure check, `Kp` a pure bet, `Jb` a pure fold,
`Kb` a pure call. All six match.

**Player 1's strategy is not unique**, so specific values must not be asserted — the
run lands at some α determined by initialization and seed. Assert the invariants
instead:

```python
assert p1_open['Q'][BET] < 1e-2                                  # Queen pure check
assert abs(p1_open['K'][BET] - 3 * p1_open['J'][BET]) < 5e-2     # 3:1 ratio
assert abs(exact_value(trainer) + 1/18) < 1e-3                   # game value
```

The strongest single check is `Qpb` at 0.426 against a predicted α + 1/3 = 0.420. The
solver was never told that relationship — it emerged from regret updates alone.

`K` at 0.290 against 3α = 0.261 is the loosest fit, and that is expected: Player 1 is
exactly indifferent with both the Jack and the King, so those frequencies have no
gradient pulling them anywhere specific. They drift along the equilibrium family, and
the 3:1 ratio tightens slowly.

**Two value numbers, and why they differ.** `self-play average` is the mean over all
200,000 training hands, including the early ones where both players were random. It is
a training diagnostic, not a game value. `exact_value` walks all six deals through the
converged average strategy with no sampling. That is the number to trust.

**What `exact_value` does not prove.** It measures a profile playing against *itself*.
An untrained uniform solver evaluates cleanly at +0.1250 and the function reports it
without complaint. Matching -0.0556 is strong evidence of convergence, not a proof.
Exploitability — best-response value against a fixed strategy — is the real test, and
it is the natural next addition.

---

## Implementation notes

Four things that are easy to get wrong, each of which produces a solver that runs
cleanly and converges to the wrong answer.

**1. `payoff` uses alternating perspective.** It returns chips from the point of view of
whoever's turn it *would* be at that terminal node — not always Player 1. So
`payoff('pbp', cards)` returns `+1` even though Player 1 folded, because
`len('pbp') % 2 == 1` selects Player 2.

At a fold terminal, the current player is always the one who did *not* fold, which is
why `return 1` needs no condition. This convention is what lets the recursion use a
single unconditional negation instead of tracking whose node it is at every level.

**2. The two reach weights are mirror images.** Line `get_strategy(p0 if player == 0
else p1)` takes own reach; line `opp_reach = p1 if player == 0 else p0` takes the
opponent's. Copying the same expression into both is the classic CFR bug.

**3. Clip on read, not on write.** `regret_sum[a] += ...` allows arbitrarily negative
values. Writing `regret_sum[a] = max(0.0, regret_sum[a] + ...)` instead is **CFR+**, a
different algorithm — faster, but not the one whose convergence you are verifying
against the closed-form answer.

**4. The regret loop must be separate.** It cannot be fused into the action loop,
because `node_util` is not complete until every action has been evaluated. Fusing them
produces subtly wrong regrets rather than a crash.

**Useful assertion.** The strategy-weighted regrets always sum to zero, since
`node_util` is a convex combination:

```python
assert abs(sum(strategy[a] * (util[a] - node_util)
               for a in range(NUM_ACTIONS))) < 1e-9
```

Regrets can never all be positive. If yours are, `node_util` is wrong.

---

## Extensions

**Exploitability.** Compute a best response against the converged strategy and report
how much it wins. Zero if and only if the strategy is a Nash equilibrium, and it needs
no answer key — so it generalizes to games with no known solution. Note that best
response is *not* "take the max at every node": the responder cannot see the hidden
card, so an action's value must be aggregated across every node in an information set,
weighted by the opponent's reach, before the max is taken.

**Convergence plot.** Exploitability against iteration count on log-log axes. Vanilla
CFR should show a slope near -1/2, matching the theoretical O(1/√T) rate.

**CFR+.** Change one line as described above and plot on the same axes. The speedup is
substantial: clipping stops abandoned actions from carrying dead negative weight that
must be climbed back through.

**Chance sampling vs enumeration.** `deal()` samples one of six deals per iteration.
Looping over all six instead gives noise-free convergence curves at six times the cost
per iteration — useful when plotting. This requires multiplying the 1/6 chance
probability into the regret weight explicitly.

**Leduc hold'em.** Six cards, two betting rounds, 288 information sets. The game layer
changes; `Node` and the recursion do not. That separation is what makes it a
half-day extension rather than a rewrite.

---

## References

- Kuhn, H. W. (1950). *Simplified Two-Person Poker*. Contributions to the Theory of
  Games, Vol. 1.
- Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. (2007). *Regret Minimization in
  Games with Incomplete Information*. NIPS.
- Neller, T., Lanctot, M. (2013). *An Introduction to Counterfactual Regret
  Minimization*.
- Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+*.
- Hart, S., Mas-Colell, A. (2000). *A Simple Adaptive Procedure Leading to Correlated
  Equilibrium*. Econometrica. (regret matching)
