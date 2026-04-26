from __future__ import annotations

from vibescents.schemas import ContextInput

_EVENT_PHRASES: dict[str, str] = {
    "Gala": "black tie gala formal event elegant luxury",
    "Date Night": "romantic date night intimate dinner sensual",
    "Casual": "casual daytime relaxed weekend effortless",
    "Business": "professional business work meeting polished",
    "Wedding": "wedding celebration elegant floral ceremony",
    "Festival": "outdoor music festival energetic lively vibrant",
}

_TIME_PHRASES: dict[str, str] = {
    "Morning": "fresh morning crisp awakening daytime",
    "Afternoon": "bright afternoon warm sunlit daytime",
    "Evening": "evening twilight dinner sophisticated",
    "Night": "late night dark mysterious nocturnal",
}

_MOOD_PHRASES: dict[str, str] = {
    "Bold": "bold statement powerful strong projection commanding",
    "Subtle": "subtle quiet understated soft delicate skin-close",
    "Fresh": "fresh clean light airy citrus aquatic",
    "Warm": "warm cozy intimate sensual gourmand amber",
    "Mysterious": "mysterious dark seductive complex smoky oud",
}


def context_to_query_string(ctx: ContextInput) -> str:
    """Assemble a natural-language query string from a ContextInput."""
    parts: list[str] = []

    if ctx.eventType:
        parts.append(_EVENT_PHRASES.get(ctx.eventType, ctx.eventType))
    if ctx.timeOfDay:
        parts.append(_TIME_PHRASES.get(ctx.timeOfDay, ctx.timeOfDay))
    if ctx.mood:
        parts.append(_MOOD_PHRASES.get(ctx.mood, ctx.mood))
    if ctx.customNotes:
        parts.append(ctx.customNotes.strip())

    if not parts:
        return "elegant versatile fragrance suitable for any occasion"

    return " | ".join(parts)
