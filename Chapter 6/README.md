# Decisions Under Uncertainty

This chapter introduces **decision theory** - how to choose actions optimally when outcomes are uncertain.

---

## 1 - Utility

How do we make rational decisions? **Utility theory** provides the foundation.

**Von Neumann-Morgenstern Axioms:**
If preferences satisfy completeness, transitivity, continuity, and independence, then there exists a **utility function** such that we should choose actions to maximize **expected utility**.

**Risk attitudes:**
- **Risk-neutral**: Care only about expected value
- **Risk-averse**: Prefer certain outcomes over gambles with same expected value
- **Risk-seeking**: Prefer gambles over certain outcomes

For example, most people are risk-averse about money: they'd rather have $50 for sure than a 50/50 chance at $0 or $100, even though both have the same expected value.

The shape of the utility function encodes risk attitudes: concave for risk-averse, linear for risk-neutral, convex for risk-seeking. This explains why people buy insurance (pay to reduce risk) and lottery tickets (pay for small chances of big rewards).

---

## 2 - Decision Networks, Value of Information, Irrationality

**Decision networks** (influence diagrams) extend Bayesian networks with decision nodes (squares) for actions we can choose and utility nodes (diamonds) for rewards/costs.

To find the optimal decision: for each possible action, infer the resulting probability distribution over outcomes, compute expected utility, and choose the action with maximum expected utility.

**Value of Information (VOI):**

Sometimes we can gather information before deciding. The value of information is the improvement in expected utility from knowing something before you act. For example: Should you check the weather forecast before deciding whether to bring an umbrella?

The notebook demonstrates this with a bus decision problem: should you wait for the bus or walk? Checking a rain forecast has positive value if it helps you make a better decision.

**Key insights:**
- Information is never harmful (in expectation)
- Information has no value if it doesn't change your decision
- VOI can guide which sensors/experiments to use

**Irrationality:**

The notebook also touches on how real humans violate the axioms of rational decision-making, such as framing effects, the sunk cost fallacy, and overestimating rare events.
