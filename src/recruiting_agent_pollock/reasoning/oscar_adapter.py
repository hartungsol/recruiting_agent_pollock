"""
OSCAR reasoner adapter.

Provides integration with an OSCAR/Pollock-style defeasible reasoner,
translating between the interview domain and the reasoning formalism.
"""

from abc import ABC, abstractmethod
from typing import Any

import httpx

from recruiting_agent_pollock.config import get_settings
from recruiting_agent_pollock.reasoning.rules import (
    Argument,
    ReasoningResult,
    RuleSet,
)


class OSCARAdapterBase(ABC):
    """Abstract base class for OSCAR reasoner adapters."""

    @abstractmethod
    async def reason(
        self,
        facts: set[str],
        rules: RuleSet,
    ) -> ReasoningResult:
        """
        Perform defeasible reasoning over facts and rules.

        Args:
            facts: Initial facts/premises.
            rules: Rules to apply.

        Returns:
            Reasoning result with conclusions and argument graph.
        """
        ...

    @abstractmethod
    async def check_defeat(
        self,
        argument: Argument,
        candidate_defeaters: list[Argument],
    ) -> list[Argument]:
        """
        Check which arguments defeat the given argument.

        Args:
            argument: The argument to check.
            candidate_defeaters: Potential defeating arguments.

        Returns:
            List of arguments that successfully defeat the input.
        """
        ...


class OSCARAdapter(OSCARAdapterBase):
    """
    Adapter for OSCAR-style defeasible reasoning.

    Communicates with an external OSCAR reasoner service to perform
    defeasible reasoning for candidate evaluation.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Initialize the OSCAR adapter.

        Args:
            endpoint: OSCAR service endpoint (uses config if not provided).
            timeout: Request timeout in seconds (uses config if not provided).
        """
        settings = get_settings()
        self._endpoint = endpoint or settings.oscar_endpoint
        self._timeout = timeout or settings.oscar_timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._endpoint,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def reason(
        self,
        facts: set[str],
        rules: RuleSet,
    ) -> ReasoningResult:
        """
        Perform defeasible reasoning over facts and rules.

        Args:
            facts: Initial facts/premises.
            rules: Rules to apply.

        Returns:
            Reasoning result with conclusions and argument graph.
        """
        # TODO: Implement OSCAR API integration
        # 1. Convert facts and rules to OSCAR format
        # 2. Send reasoning request
        # 3. Parse response into ReasoningResult

        # Placeholder implementation using local reasoning
        return await self._local_reason(facts, rules)

    async def _local_reason(
        self,
        facts: set[str],
        rules: RuleSet,
    ) -> ReasoningResult:
        """
        Perform simple local reasoning when OSCAR is not available.

        This is a simplified fallback that doesn't implement full
        defeasible reasoning semantics.

        Args:
            facts: Initial facts/premises.
            rules: Rules to apply.

        Returns:
            Simplified reasoning result.
        """
        # Simple forward chaining (not true defeasible reasoning)
        conclusions: list[str] = []
        arguments: list[Argument] = []
        reasoning_trace: list[dict[str, Any]] = []

        current_facts = facts.copy()
        changed = True

        while changed:
            changed = False
            applicable = rules.get_applicable_rules(current_facts)

            for rule in applicable:
                if rule.consequent not in current_facts:
                    current_facts.add(rule.consequent)
                    conclusions.append(rule.consequent)

                    argument = Argument(
                        premises=rule.antecedent,
                        conclusion=rule.consequent,
                        rules_applied=[rule.rule_id],
                        strength=rule.strength,
                    )
                    arguments.append(argument)

                    reasoning_trace.append({
                        "step": "rule_application",
                        "rule": rule.name,
                        "antecedent": rule.antecedent,
                        "consequent": rule.consequent,
                    })

                    changed = True

        return ReasoningResult(
            conclusions=conclusions,
            arguments=arguments,
            defeated_arguments=[],
            defeater_graph=[],
            reasoning_trace=reasoning_trace,
        )

    async def check_defeat(
        self,
        argument: Argument,
        candidate_defeaters: list[Argument],
    ) -> list[Argument]:
        """
        Check which arguments defeat the given argument.

        Args:
            argument: The argument to check.
            candidate_defeaters: Potential defeating arguments.

        Returns:
            List of arguments that successfully defeat the input.
        """
        # TODO: Implement proper defeat checking via OSCAR
        # For now, simple strength-based comparison

        defeaters = []
        for candidate in candidate_defeaters:
            # Check if candidate contradicts or undercuts argument
            if candidate.strength > argument.strength:
                # Simplified: stronger argument defeats weaker
                defeaters.append(candidate)

        return defeaters

    async def evaluate_candidate_claim(
        self,
        claim: str,
        supporting_evidence: list[str],
        contradicting_evidence: list[str],
    ) -> dict[str, Any]:
        """
        Evaluate a candidate claim using defeasible reasoning.

        Args:
            claim: The claim to evaluate.
            supporting_evidence: Evidence supporting the claim.
            contradicting_evidence: Evidence against the claim.

        Returns:
            Evaluation result with reasoning.
        """
        # TODO: Build arguments from evidence and evaluate

        # Placeholder implementation
        support_strength = len(supporting_evidence) * 0.2
        contra_strength = len(contradicting_evidence) * 0.2

        if support_strength > contra_strength:
            status = "supported"
            confidence = min(0.9, support_strength - contra_strength + 0.5)
        elif contra_strength > support_strength:
            status = "defeated"
            confidence = min(0.9, contra_strength - support_strength + 0.5)
        else:
            status = "undetermined"
            confidence = 0.5

        return {
            "claim": claim,
            "status": status,
            "confidence": confidence,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
        }

    async def propose_defeaters(
        self,
        *,
        hypothesis_statement: str,
        basis: list[str] | None = None,
        experience_level: str = "unknown",
        max_defeaters: int = 3,
    ) -> list[str]:
        """Propose defeaters for an *unconfirmed* hypothesis.

        This is intentionally conservative and uses a local fallback.
        In a future version, this can call an external OSCAR service.
        """
        basis = basis or []

        defeaters: list[str] = []

        # General undercutter: absence of evidence is not evidence of absence.
        if "not mentioned" in hypothesis_statement.lower():
            defeaters.append("Not yet discussed; candidate may still have this experience.")

        # General rebutter: ask for concrete evidence.
        defeaters.append("Requires confirmation with a concrete example and follow-up questions.")

        # Experience-sensitive undercutter: juniors may omit details.
        if experience_level == "entry":
            defeaters.append("Entry-level candidates may omit details; probe projects/coursework before inferring a gap.")
        elif experience_level == "senior":
            defeaters.append("Senior candidates may summarize; ask for a specific scenario before inferring a gap.")

        # Avoid duplicates, cap.
        unique: list[str] = []
        seen: set[str] = set()
        for d in defeaters:
            dl = d.lower().strip()
            if dl and dl not in seen:
                unique.append(d)
                seen.add(dl)
        return unique[:max_defeaters]
