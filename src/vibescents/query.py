from __future__ import annotations

from vibescents.schemas import ContextInput

_EVENT_PHRASES: dict[str, str] = {
    "Gala": "black tie gala formal event elegant luxury oriental, chypre, oud, amber, incense, opulent, sillage",
    "Date Night": "romantic date night intimate dinner sensual",
    "Casual": "casual daytime relaxed weekend effortless",
    "Business": "professional business work meeting polished",
    "Wedding": "wedding celebration elegant floral ceremony",
    "Festival": "outdoor music festival energetic lively vibrant",
}

_TIME_PHRASES: dict[str, str] = {
    "Morning": "fresh morning crisp awakening daytime citrus, green, bergamot, light projection",
    "Afternoon": "bright afternoon warm sunlit daytime",
    "Evening": "evening twilight dinner sophisticated",
    "Night": "late night dark mysterious nocturnal",
}

_MOOD_PHRASES: dict[str, str] = {
    "Bold": "bold statement powerful strong projection commanding oud, leather, intense, high projection",
    "Subtle": "subtle quiet understated soft delicate skin-close",
    "Fresh": "fresh clean light airy citrus aquatic citrus, aquatic, green, bergamot, petrichor",
    "Warm": "warm cozy intimate sensual gourmand amber vanilla, amber, gourmand, musk, tonka",
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


def build_candidate_text(row: object) -> str:
    '''Build rich reranker input from an enriched corpus row.'''
    parts = []
    name = _str_field(row, 'name')
    brand = _str_field(row, 'brand')
    if name:
        label = f'{name} by {brand}' if brand else name
        parts.append(label)
    vibe = _str_field(row, 'vibe_sentence')
    if vibe:
        parts.append(vibe)
    occasion = _str_field(row, 'likely_occasion')
    if occasion:
        parts.append(f'Occasion: {occasion}')
    notes_raw = ' | '.join(filter(None, [
        _str_field(row, 'top_notes'),
        _str_field(row, 'middle_notes'),
        _str_field(row, 'base_notes'),
    ]))
    if notes_raw:
        parts.append(f'Notes: {notes_raw}')
    return ' | '.join(parts) if parts else 'fragrance'


def _str_field(row: object, key: str) -> str:
    '''Extract a clean string from a row dict-like or pandas Series.'''
    try:
        val = row[key] if hasattr(row, '__getitem__') else getattr(row, key, None)
    except (KeyError, IndexError):
        return ''
    if val is None:
        return ''
    s = str(val).strip()
    return '' if s.lower() in ('nan', 'none', '') else s
