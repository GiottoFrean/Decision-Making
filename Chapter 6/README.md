# Decisions Under Uncertainty

Moving from inference to action, this chapter introduces **decision theory** - how to choose actions optimally when outcomes are uncertain.

---

## 1 - Utility

How do we make rational decisions? **Utility theory** provides the foundation.

**Von Neumann-Morgenstern Axioms:**
If preferences satisfy:
- **Completeness**: Can compare any two options
- **Transitivity**: If A ≻ B and B ≻ C, then A ≻ C
- **Continuity**: No option is infinitely better or worse
- **Independence**: Preferences between lotteries don't depend on irrelevant alternatives

Then there exists a **utility function** U such that we should choose actions to maximize **expected utility**:

$$EU(a|o) = \sum_{s'} P(s'|a, o) \cdot U(s')$$

**Risk attitudes:**
- **Risk-neutral**: Care only about expected value
- **Risk-averse**: Prefer certain outcomes over gambles with same expected value
- **Risk-seeking**: Prefer gambles over certain outcomes

For example, most people are risk-averse about money: they'd rather have \\$50 for sure than a 50/50 chance at \\$0 or \\$100, even though both have the same expected value.

The shape of the utility function encodes risk attitudes:
- Concave: risk-averse
- Linear: risk-neutral
- Convex: risk-seeking

This explains why people buy insurance (pay to reduce risk) and lottery tickets (pay for small chances of big rewards).

---

## 2 - Decision Networks, Value of Information, Irrationality

**Decision networks** (influence diagrams) extend Bayesian networks with:
- **Decision nodes** (squares): Actions we can choose
- **Utility nodes** (diamonds): Rewards/costs

To find the optimal decision:
1. For each possible action
2. Infer the resulting probability distribution over outcomes
3. Compute expected utility
4. Choose the action with maximum expected utility

**Value of Information (VOI):**

Sometimes we can gather information before deciding. Is it worth it?

$$VOI = EU(\text{with information}) - EU(\text{without information})$$

For example: Should you check the weather forecast before deciding whether to bring an umbrella? The VOI is the improvement in expected utility from knowing the forecast.

The notebook demonstrates this with a bus decision problem: should you wait for the bus or walk? Checking a rain forecast has positive VOI if it helps you make a better decision.

**Key insights:**
- Information is never harmful (in expectation) - VOI ≥ 0
- Information has no value if it doesn't change your decision
- VOI can guide which sensors/experiments to use

**Irrationality:**

The notebook also touches on how real humans violate these axioms:
- Framing effects (same choice described differently leads to different decisions)
- Sunk cost fallacy
- Probability weighting (overestimating rare events)

Understanding normative decision theory helps us build better AI systems and understand human biases.
