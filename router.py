"""Model registry and automatic routing.

The router looks at the request (images, answer mode, academic level, wording,
length) and picks a *designation* (`fast`, `general`, `reasoning`) plus a ranked
list of candidate models. Vision is a hard requirement when images are present.
The orchestrator tries candidates in order, so a rate-limited or failing
provider automatically falls back to the next one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .base import AIProvider, ModelSpec, NoModelAvailable

_REASONING_MODES = {"deep-reasoning", "research"}
_HARD_LEVELS = {"postgraduate", "phd", "research", "research-level", "graduate", "masters", "doctoral"}
_EASY_LEVELS = {"elementary", "middle-school", "primary"}
_HARD_WORDS = re.compile(
    r"\b(prove|proof|derive|derivation|theorem|lemma|corollary|integral|differential equation|eigen\w*|asymptotic\w*|"
    r"hamiltonian|lagrangian|manifold|topolog\w*|stochastic|converg\w+|np-hard|quantum|relativity|thermodynamic\w*|"
    r"tensor|fourier|laplace|bayesian|regression|optimi[sz]ation|cryptograph\w*)\b|[∫∑∏∂∇]",
    re.I,
)

# Preference order of designations for each role.
_WANT = {
    "reasoning": ("reasoning", "general", "fast", "vision"),
    "general": ("general", "reasoning", "fast", "vision"),
    "fast": ("fast", "general", "reasoning", "vision"),
}


@dataclass
class RouteInput:
    question: str
    mode: str
    level: str
    has_images: bool
    requested_model: str = "auto"


@dataclass
class RoutingDecision:
    designation: str
    need_vision: bool
    reasoning_effort: Optional[str]
    reasons: list[str] = field(default_factory=list)
    candidates: list[ModelSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, providers: list[AIProvider], priority: list[str]):
        self.providers = {p.name: p for p in providers}
        self.priority = priority
        self.specs: list[ModelSpec] = [s for p in providers for s in p.specs]
        self._by_key = {s.key: s for s in self.specs}

    def get(self, key: str) -> Optional[ModelSpec]:
        return self._by_key.get(key)

    def provider_for(self, spec: ModelSpec) -> AIProvider:
        return self.providers[spec.provider]

    def _priority_index(self, spec: ModelSpec) -> int:
        try:
            return self.priority.index(spec.provider)
        except ValueError:
            return len(self.priority)

    def rank(self, designation: str, need_vision: bool) -> list[ModelSpec]:
        want = _WANT.get(designation, _WANT["general"])

        def score(spec: ModelSpec) -> tuple[int, int, int]:
            best = min((want.index(d) for d in spec.designations if d in want), default=len(want))
            vision_pref = 0 if (need_vision and "vision" in spec.designations) else 1
            return best, vision_pref, self._priority_index(spec)

        pool = [s for s in self.specs if (s.vision or not need_vision)]
        return sorted(pool, key=score)


class Router:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @staticmethod
    def assess(inp: RouteInput) -> tuple[str, Optional[str], list[str]]:
        """Return (designation, reasoning_effort, reasons). Transparent heuristics, easy to tune."""
        score, reasons = 0, []
        mode, level, q = inp.mode.lower(), inp.level.lower(), inp.question
        if mode in _REASONING_MODES:
            score += 3
            reasons.append(f"answer mode '{mode}' asks for deeper reasoning")
        if level in _HARD_LEVELS:
            score += 2
            reasons.append(f"academic level '{level}'")
        hits = len(_HARD_WORDS.findall(q))
        if hits:
            score += min(hits, 2)
            reasons.append("advanced technical wording")
        if len(q) > 1500:
            score += 1
            reasons.append("long, multi-part question")

        if score >= 3:
            effort = "high" if (mode == "deep-reasoning" or level in {"phd", "research", "research-level"}) else "medium"
            return "reasoning", effort, reasons
        if score == 0 and mode == "short" and len(q) < 400 and not inp.has_images:
            return "fast", "low", ["short answer to a short question"]
        return "general", "medium" if score else "low", reasons or ["standard question"]

    def decide(self, inp: RouteInput) -> RoutingDecision:
        designation, effort, reasons = self.assess(inp)
        decision = RoutingDecision(designation, inp.has_images, effort, reasons)
        if inp.has_images:
            decision.reasons.append("images need a vision-capable model")

        ranked = self.registry.rank(designation, inp.has_images)
        requested = (inp.requested_model or "auto").strip()
        if requested.lower() != "auto":
            spec = self.registry.get(requested)
            if spec is None:
                decision.warnings.append(f"Model '{requested}' is not available; automatic routing was used.")
            elif inp.has_images and not spec.vision:
                decision.warnings.append(f"Model '{requested}' cannot read images; automatic routing was used.")
            else:
                ranked = [spec] + [s for s in ranked if s.key != spec.key]
                decision.reasons.insert(0, "model chosen by the user")

        if not ranked:
            if inp.has_images:
                raise NoModelAvailable(
                    "No vision-capable model is configured. Set <PROVIDER>_VISION=true or <PROVIDER>_VISION_MODEL."
                )
            raise NoModelAvailable("No AI model is configured. Add a provider API key and model to backend/.env.")
        decision.candidates = ranked
        return decision
