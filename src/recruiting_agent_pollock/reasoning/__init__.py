"""
Reasoning module for defeasible logic integration.

Provides integration with OSCAR/Pollock-style defeasible reasoners
for nuanced decision-making in candidate evaluation.
"""

from recruiting_agent_pollock.reasoning.oscar_adapter import OSCARAdapter
from recruiting_agent_pollock.reasoning.hypotheses import Hypothesis, HypothesisStatus
from recruiting_agent_pollock.reasoning.rules import (
    Argument,
    Defeater,
    ReasoningResult,
    Rule,
    RuleSet,
)

__all__ = [
    "OSCARAdapter",
    "Hypothesis",
    "HypothesisStatus",
    "Rule",
    "RuleSet",
    "Argument",
    "Defeater",
    "ReasoningResult",
]
