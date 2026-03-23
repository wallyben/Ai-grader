"""
Price Ingestion Layer — Phase 2
Supports manual price input and returns structured pricing data.
Optional scraping stub included — does NOT rely on paid APIs.

All prices are in the same currency (user-supplied, e.g. EUR or USD).
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ─── Typical PSA premium multipliers (conservative estimates) ─────────────────
# These are fallback estimates when actual market prices are unknown.
# Based on common TCG/sports card market patterns.
_PSA_8_MULTIPLIER  = 2.0
_PSA_9_MULTIPLIER  = 3.5
_PSA_10_MULTIPLIER = 8.0


@dataclass
class CardPricing:
    """Structured pricing data for a single card."""
    card_name:  str
    raw_price:  float
    psa_8:      Optional[float] = None
    psa_9:      Optional[float] = None
    psa_10:     Optional[float] = None
    source:     str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_complete(self) -> bool:
        """True if all three PSA grade prices are provided."""
        return all(x is not None for x in [self.psa_8, self.psa_9, self.psa_10])

    def fill_estimates(self) -> "CardPricing":
        """
        Return a new CardPricing with missing PSA prices filled using
        conservative multiplier estimates. Source marked as 'estimated'
        if any prices were missing.
        """
        raw = self.raw_price
        any_missing = not self.is_complete()
        return CardPricing(
            card_name=self.card_name,
            raw_price=raw,
            psa_8=self.psa_8   if self.psa_8  is not None else round(raw * _PSA_8_MULTIPLIER,  2),
            psa_9=self.psa_9   if self.psa_9  is not None else round(raw * _PSA_9_MULTIPLIER,  2),
            psa_10=self.psa_10 if self.psa_10 is not None else round(raw * _PSA_10_MULTIPLIER, 2),
            source="estimated" if any_missing else self.source,
        )


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_pricing(
    card_name:  str,
    raw_price:  float,
    psa_8:      Optional[float] = None,
    psa_9:      Optional[float] = None,
    psa_10:     Optional[float] = None,
    grading_cost: Optional[float] = None,
) -> "CardPricingBundle":
    """
    Create a CardPricingBundle from manual inputs.
    grading_cost is stored separately (it is a cost, not a market price).
    """
    pricing = CardPricing(
        card_name=card_name,
        raw_price=raw_price,
        psa_8=psa_8,
        psa_9=psa_9,
        psa_10=psa_10,
        source="manual",
    )
    return CardPricingBundle(pricing=pricing, grading_cost=grading_cost or 0.0)


@dataclass
class CardPricingBundle:
    """Pricing + grading cost bundle passed into the ROI engine."""
    pricing:      CardPricing
    grading_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.pricing.to_dict(),
            "grading_cost": self.grading_cost,
        }


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_pricing(pricing: CardPricing) -> Dict[str, Any]:
    """
    Validate pricing data and return a validation report.
    Checks logical consistency: psa_8 <= psa_9 <= psa_10, positive values.
    """
    issues: List[str]   = []
    warnings: List[str] = []

    if pricing.raw_price <= 0:
        issues.append("raw_price must be positive")

    if pricing.psa_8 is not None and pricing.psa_8 < pricing.raw_price:
        warnings.append("PSA 8 price is lower than raw — unusual but may reflect low demand for slabs")

    if pricing.psa_8 is not None and pricing.psa_8 <= 0:
        issues.append("psa_8 must be positive")

    if pricing.psa_9 is not None and pricing.psa_9 <= 0:
        issues.append("psa_9 must be positive")

    if pricing.psa_10 is not None and pricing.psa_10 <= 0:
        issues.append("psa_10 must be positive")

    if pricing.psa_9 is not None and pricing.psa_8 is not None:
        if pricing.psa_9 < pricing.psa_8:
            issues.append("PSA 9 price cannot be lower than PSA 8 price")

    if pricing.psa_10 is not None and pricing.psa_9 is not None:
        if pricing.psa_10 < pricing.psa_9:
            issues.append("PSA 10 price cannot be lower than PSA 9 price")

    return {
        "valid":    len(issues) == 0,
        "issues":   issues,
        "warnings": warnings,
    }


# ─── Scraping stub ─────────────────────────────────────────────────────────────
# Implement here if/when external price fetching is needed.
# Do NOT use paid APIs. Respect ToS of any site scraped.

def fetch_market_prices(card_name: str) -> Optional[CardPricing]:
    """
    Stub for optional market price scraping.
    Returns None (not implemented).

    To implement: scrape eBay completed listings, Cardmarket, etc.
    Must respect rate limits and terms of service.
    """
    return None
