from __future__ import annotations

import re

_SPACE = re.compile(r"\s+")
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")
_LEAD = re.compile(r"^(?:>>\s*|[-–—]\s*)+")
_WRAPPED = re.compile(r"^[\(\[][^\)\]]*[\)\]]?[.!?]*$")
_SFX = re.compile(
    r"silence|inaudible|blank.?audio|end of audio|non-?english|"
    r"birds?|insects?|crickets?|keyboard|typing|clicking|clacking|"
    r"clears?\s+throat|sighs?|footsteps?|door|applause|clapping|"
    r"lips|cough(?:ing)?|laugh(?:ing)?|indistinct|music|clippers?|"
    r"speaking(?:\s+in)?(?:\s+a)?\s+foreign\s+language|"
    r"sniff|static|wind|water|audience",
    re.I,
)
_THANKS = re.compile(
    r"^(?:thank(?:s| you)(?:\s+(?:you|very much|so much))?)"
    r"(?:\s*[,.]?\s*thank(?:s| you)(?:\s+(?:you|very much|so much))?)*"
    r"(?:\s*,?\s*and have a good time)?[.!?]*$",
    re.I,
)
_FILLER = re.compile(
    r"thank(?:s| you) for watching|"
    r"thanks for watching|"
    r"thanks for having (?:time|me)|"
    r"thank you for your attention|"
    r"thanks for listening|"
    r"peace[,.]?\s*god bless|"
    r"(?:i(?:'ll| will)|we(?:'ll| will)|let'?s)\s+see you|"
    r"see you(?:\s+guys)?\s+(?:next time|later|soon|tomorrow|in the next)\b|"
    r"until next time|"
    r"catch you later|"
    r"like and subscribe|"
    r"don't forget to subscribe|"
    r"that's all for (?:today|now|this)|"
    r"^(?:bye(?:[\s-]*bye)?|good[\s-]*bye)[.!?]*$|"
    r"^cheers[.!?]*$|"
    r"^(?:right|all\s*right|alright)[.!?]*$",
    re.I,
)
GREET = re.compile(
    r"^(hello|hi|hey|thanks?|thank you|okay|ok|yeah|yes|no|mm-?hmm|"
    r"right|all\s*right|alright)[.!?]*$",
    re.I,
)
_TAG = re.compile(r"non-?english|foreign language|inaudible|end of audio", re.I)
_BLANK = frozenset({"", "[blank_audio]", "blank_audio"})
_TINY = frozenset({".", "you", "*sniff*"})


def english_only(text: str) -> str:
    """Keep ASCII + CJK; drop other scripts that usually come from ASR noise."""
    return _SPACE.sub(" ", re.sub(r"[^\x00-\x7F\u4e00-\u9fff]+", " ", text)).strip()


def _norm(text: str) -> str:
    return _LEAD.sub("", _SPACE.sub(" ", text.strip()))


def _is_junk(t: str) -> bool:
    low = t.lower()
    if low in _BLANK or low in _TINY:
        return True
    if _WRAPPED.match(t) or _THANKS.match(t) or _FILLER.search(t) or _TAG.search(t):
        return True
    if t[:1] in "([" and _SFX.search(t):
        return True
    n = 0
    for c in t:
        if "A" <= c <= "Z" or "a" <= c <= "z" or "\u4e00" <= c <= "\u9fff":
            n += 1
            if n >= 3:
                return False
    return True


def is_junk_line(text: str) -> bool:
    """Drop Whisper SFX, non-English, and filler lines from the transcript."""
    return _is_junk(_norm(text))


def is_skip_ext(text: str) -> bool:
    t = _norm(text)
    return (not t) or _is_junk(t) or bool(GREET.match(t))
