"""
Data structures for defeasible reasoning rules.

Defines the building blocks for constructing defeasible arguments
used by the OSCAR reasoner.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RuleType(str, Enum):
    """Types of reasoning rules."""

    PRIMA_FACIE = "prima_facie"  # Defeasible rule
    CONCLUSIVE = "conclusive"  # Non-defeasible rule
    REBUTTING = "rebutting"  # Defeats by contradicting conclusion
    UNDERCUTTING = "undercutting"  # Defeats by attacking inference


class Rule(BaseModel):
    """
    A defeasible reasoning rule.

    Rules can be prima facie (defeasible) or conclusive (non-defeasible),
    and can serve as rebutting or undercutting defeaters.
    """

    rule_id: UUID = Field(default_factory=uuid4, description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    rule_type: RuleType = Field(..., description="Type of rule")
    antecedent: list[str] = Field(
        default_factory=list,
        description="Conditions that must hold for rule to apply",
    )
    consequent: str = Field(..., description="Conclusion when rule fires")
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the inference (for prima facie rules)",
    )
    description: str = Field(default="", description="Human-readable description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Argument(BaseModel):
    """
    An argument in defeasible reasoning.

    Represents a chain of reasoning from premises to a conclusion,
    which may be defeated by other arguments.
    """

    argument_id: UUID = Field(default_factory=uuid4, description="Unique argument identifier")
    premises: list[str] = Field(..., description="Starting premises")
    conclusion: str = Field(..., description="Derived conclusion")
    rules_applied: list[UUID] = Field(
        default_factory=list,
        description="Rules used in this argument",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall argument strength",
    )
    is_defeated: bool = Field(default=False, description="Whether argument is defeated")
    defeaters: list[UUID] = Field(
        default_factory=list,
        description="Arguments that defeat this one",
    )


class Defeater(BaseModel):
    """
    A defeater relationship between arguments.

    Represents how one argument defeats another, either by
    rebutting (contradicting) or undercutting (attacking inference).
    """

    defeater_id: UUID = Field(default_factory=uuid4, description="Unique defeater identifier")
    defeating_argument: UUID = Field(..., description="The defeating argument")
    defeated_argument: UUID = Field(..., description="The defeated argument")
    defeater_type: RuleType = Field(..., description="Type of defeat (rebutting/undercutting)")
    strength_comparison: float = Field(
        default=0.0,
        description="Relative strength difference",
    )


class RuleSet(BaseModel):
    """
    A collection of reasoning rules for a domain.

    Organizes rules and provides methods for rule lookup and
    applicability checking.
    """

    name: str = Field(..., description="Name of the rule set")
    description: str = Field(default="", description="Description of the rule set")
    rules: list[Rule] = Field(default_factory=list, description="Rules in this set")
    priority_ordering: list[UUID] = Field(
        default_factory=list,
        description="Priority ordering of rules (higher priority first)",
    )

    def get_applicable_rules(self, facts: set[str]) -> list[Rule]:
        """
        Get rules whose antecedents are satisfied by the given facts.

        Args:
            facts: Set of known facts.

        Returns:
            List of applicable rules.
        """
        applicable = []
        for rule in self.rules:
            if all(ant in facts for ant in rule.antecedent):
                applicable.append(rule)
        return applicable

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the set.

        Args:
            rule: Rule to add.
        """
        self.rules.append(rule)


class ReasoningResult(BaseModel):
    """
    Result of a reasoning process.

    Contains the conclusions reached, the arguments supporting them,
    and any defeated arguments.
    """

    conclusions: list[str] = Field(
        default_factory=list,
        description="Undefeated conclusions",
    )
    arguments: list[Argument] = Field(
        default_factory=list,
        description="All arguments constructed",
    )
    defeated_arguments: list[Argument] = Field(
        default_factory=list,
        description="Arguments that were defeated",
    )
    defeater_graph: list[Defeater] = Field(
        default_factory=list,
        description="Defeater relationships",
    )
    reasoning_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Step-by-step trace of reasoning",
    )


# Predefined recruiting-specific rules
RECRUITING_RULES = RuleSet(
    name="recruiting_evaluation",
    description="Rules for evaluating recruiting candidates",
    rules=[
        Rule(
            name="skill_match_positive",
            rule_type=RuleType.PRIMA_FACIE,
            antecedent=["candidate_has_required_skill"],
            consequent="candidate_is_qualified",
            strength=0.7,
            description="Having a required skill is prima facie evidence of qualification",
        ),
        Rule(
            name="experience_gap",
            rule_type=RuleType.PRIMA_FACIE,
            antecedent=["candidate_lacks_experience_years"],
            consequent="candidate_may_be_underqualified",
            strength=0.5,
            description="Lacking experience is prima facie evidence of being underqualified",
        ),
        Rule(
            name="demonstrated_learning",
            rule_type=RuleType.UNDERCUTTING,
            antecedent=["candidate_demonstrates_quick_learning"],
            consequent="defeats(candidate_may_be_underqualified)",
            strength=0.6,
            description="Demonstrated learning ability defeats experience gap concerns",
        ),
        Rule(
            name="inconsistent_claims",
            rule_type=RuleType.PRIMA_FACIE,
            antecedent=["candidate_made_inconsistent_claims"],
            consequent="candidate_credibility_concern",
            strength=0.6,
            description="Inconsistent claims raise credibility concerns",
        ),
        Rule(
            name="clarification_resolves",
            rule_type=RuleType.REBUTTING,
            antecedent=["candidate_provides_valid_clarification"],
            consequent="no_credibility_concern",
            strength=0.7,
            description="Valid clarification rebuts credibility concerns",
        ),
    ],
)
