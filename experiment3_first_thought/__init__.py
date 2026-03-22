"""
Experiment 3: First Thought vs. Reasoned Explanation

Standalone liquid/CfC implementation of the first-thought study.
A CfC observer watches a CfC executor trained on eight dynamical
systems, then is evaluated on whether immediate predictions outperform
iterative deliberation as K increases.

Core outputs:
- single-pass vs deliberated next-hidden-state prediction error
- K-sweep curves across increasing deliberation depth
- reaction-time analog using pass count as latency units
- stability/convergence analysis as a confidence analog
- trained observer vs. untrained, linear, shuffled, and wrong-executor controls
"""
