"""The safety gate — deterministic, and first in line. The centrepiece.

Crisis handling must not depend on retrieval quality, model mood, provider
uptime, or prompt adherence. So it is a fixed code path: a pattern net over
every user turn, running BEFORE the understander, retrieval and the LLM. When
it fires, generation is bypassed entirely — the caller hears a fixed, warm
handoff to human helplines, the UI takes over with tap-to-call numbers, and
the event is recorded. A prompt is a request; this is a guarantee.

Design notes, earned through the eval set (scripts/run_safety_evals.py):

- Recall beats precision here, but false positives are not free: being read a
  crisis script because you said "this deadline is killing me" is alienating.
  The patterns are INTENT-ANCHORED — "kill myself", not "killing"; "mar jaana
  chahta hoon", not the colloquial "mar gaya yaar".
- Three languages of despair: English, Hindi (Devanagari), and romanized
  Hinglish. STT may deliver any of them, including Devanagari transliterations
  of English words.
- Oblique phrasings are the hard 20%: "I don't see the point anymore",
  "everyone would be better off without me", "I don't want to wake up
  tomorrow". Each earned a pattern via the eval set, which is the contract:
  the gate must score 100% recall there before any deployment.
- Once fired, the gate stays engaged for the session: subsequent turns get a
  short deterministic reaffirmation, not a resumed coaching conversation.

The scope gate rides along: diagnosis/medication asks are detected
deterministically too, but they inject a binding decline directive rather
than bypassing the model — refusing warmly benefits from generation; crisis
handoff must not.

The third layer is the injection net. This is a voice product: attacks
arrive spoken, through STT — no unicode tricks, no base64, just the things a
person can say out loud. Three classes cover that surface: adopting a
clinical persona ("roleplay as my doctor" is a scope violation wearing a
costume), overriding instructions ("ignore your rules"), and extracting the
prompt ("what's your system prompt"). On a match the turn is deflected the
same way crisis is handled — a fixed line is spoken and the model never sees
the attempt — but nothing latches: one deflection, back to the conversation.
Precedence is crisis > injection > scope, always: "ignore your rules, I just
want to die" is a crisis, not an attack.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/nukta variants, collapse spaces.

    Keeps Devanagari intact; removes the '़' nukta so ज़िंदगी and जिंदगी match
    the same pattern.
    """
    text = unicodedata.normalize("NFC", text or "").lower()
    text = text.replace("़", "")  # nukta
    text = re.sub(r"[^\w\sऀ-ॿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Crisis net. Every pattern is here because an eval case demands it; add the
# failing case FIRST, then the pattern that catches it.
# --------------------------------------------------------------------------

_CRISIS = [
    # ---- English, explicit
    r"\bsuicid\w*",
    r"\bkill(ing)? myself\b",
    r"\bend(ing)? (my|this) life\b",
    r"\btake my (own )?life\b",
    r"\b(want|wanna|going|gonna|plan(ning)?) to die\b",
    r"\bwish i (was|were) dead\b",
    r"\bbetter off dead\b",
    r"\bbetter off without me\b",
    r"\b(hurt(ing)?|harm(ing)?|cut(ting)?) myself\b",
    r"\bself[ -]?harm\w*",
    r"\boverdos\w+",
    r"\bend it all\b",
    r"\bno reason (left )?to (live|go on)\b",
    r"\bnot worth living\b",
    r"\bdon.?t want to (live|be alive)\b",
    # ---- English, oblique — each from a real eval case
    r"\b(don.?t|can.?t) see (the|any) point( in| of)? (living|going on|anything|any of this)?\b",
    r"\bno point (in )?(anything|living|going on) ?(anymore|any more)?\b",
    r"\bwhat.?s the point ?(anymore|any more)\b",
    r"\bjust want (it all )?to (stop|end|be over)\b",
    r"\bwant to disappear\b",
    r"\b(don.?t|do not) want to wake up\b",
    r"\bsleep and (never|not) wake\b",
    r"\bcan.?t (do|take) this anymore\b.*\b(life|living|alive)\b",
    r"\beveryone would be (better|happier) without me\b",
    r"\bnobody would (miss|even miss) me\b",
    r"\btired of being alive\b",
    r"\bdone with (life|living|everything)\b",
    # ---- abuse / violence toward the caller
    r"\b(hits?|beats?|hurts?) me\b",
    r"\bbeing abused\b",
    r"\babus(es|ing) me\b",
    r"\bafraid (he|she|they).{0,20}(hurt|kill|hit)\b",
    # ---- Hindi (Devanagari)
    r"आत्महत्या",
    r"खुदकुशी",
    r"मरना (चाहत|चाहत[ीे]|है)",
    r"मर (जाना|जाऊ|जाऊं|जाना चाहत)",
    r"जीना नहीं चाहत",
    r"जान दे (दूं|दूंगा|दूंगी|देना)",
    r"जिंदगी खत्म",
    r"जीने का (कोई )?(मन|मतलब) नहीं",
    r"खुद को (चोट|नुकसान)",
    r"मारता है मुझे|मारती है मुझे|पीटता है|पीटती है",
    # ---- romanized Hinglish (intent-anchored: not the colloquial "mar gaya")
    r"\bkhudkushi\b|\baatm[ao]?hatya\b",
    r"\bmar+n[ae] (chaht|hai)\w*",
    r"\bmar ja(na|au|aun|ana) chaht\w*",
    r"\bjeena nahi\w* chaht\w*",
    r"\bjaan de (dunga|dungi|doon|dena)\b",
    r"\bzindagi khatam\b|\bsab khatam kar\w*",
    r"\bjeene ka (koi )?(man|matlab) nahi\b",
    r"\bkhud ko (chot|hurt|nuksan)\b",
    r"\bmarta hai mujhe\b|\bpeet(ta|ti) hai\b",
]
_CRISIS_RX = [re.compile(p) for p in _CRISIS]

# --------------------------------------------------------------------------
# Scope net: diagnosis / medication asks. Detection is deterministic; the
# response is a binding directive to the model (warm refusal benefits from
# generation; crisis handoff must not).
# --------------------------------------------------------------------------

_SCOPE = [
    r"\bdo (i|you think i) have\b.{0,40}\b(depress|anxiet|adhd|bipolar|ocd|ptsd|disorder|condition)\w*",
    r"\b(am i|i think i.?m)\b.{0,20}\b(depressed|bipolar|autistic|mentally ill)\b",
    r"\bdiagnos\w+ me\b|\bwhat.?s my diagnosis\b|\bdiagnose\b.{0,24}\bme\b",
    r"\b(should|can) i (take|stop|start|increase|reduce)\b.{0,30}\b(med(s|icine|ication)?|antidepressant|dose|dosage|pill)\w*",
    r"\bwhich (medicine|medication|tablet|pill|antidepressant)\b",
    r"\b(prescri|dosage|dose of)\w*",
    r"\bmujhe (depression|anxiety) (hai kya|to nahi)\b",
    r"\bkya mujhe\b.{0,30}\b(depression|anxiety|bimari)\b",
    r"(दवा|दवाई|गोली).{0,20}(लूं|लेनी|कौन सी|बंद)",
    r"क्या मुझे (डिप्रेशन|एंग्जायटी|कोई बीमारी) है",
]
_SCOPE_RX = [re.compile(p) for p in _SCOPE]

# --------------------------------------------------------------------------
# Injection net: role override, instruction override, prompt extraction.
# The same discipline as the crisis net — intent-anchored, every pattern
# earned by an eval case, false positives are not free. "My doctor told me
# to sleep more" mentions a doctor; "act as my doctor" assigns her the role.
# Verbs are matched in the imperative only ("act as", never "acts as") so
# narration about other people passes clean.
# --------------------------------------------------------------------------

_CLINICIAN = r"(doctor|daktar|physician|therapist|psychiatrist|psychologist|pharmacist|nurse|counsell?or|clinician)"

_INJECTION = [
    # ---- role override: hand her a clinical persona
    rf"\b(role ?play|pretend|imagine|act|play)\w*\b.{{0,24}}\b(you ?re|you are|yourself)\b.{{0,14}}\b(a |an |my |the )?{_CLINICIAN}\b",
    rf"\b(role ?play|act|answer|reply|respond) as (a |an |my |your )?{_CLINICIAN}\b",
    rf"\b(pretend|act) to be (a |an |my |the )?{_CLINICIAN}\b",
    rf"\byou( to)? (be|become|act as) (a |an |my |the )?{_CLINICIAN}\b",
    rf"\byou (are|re) now (a |an |my |the )?{_CLINICIAN}\b",
    # ---- role override, Hindi / Hinglish
    rf"\b{_CLINICIAN} ban(o| ?ja[ao]?)\b",
    rf"\btum (ab |ek |meri |mere )*{_CLINICIAN} (ho|bano)\b",
    r"डॉक्टर बन(ो| जा)",
    r"तुम (अब )?(एक )?डॉक्टर (हो|बनो)",
    # ---- instruction override
    r"\b(ignore|disregard|forget|bypass|override|disable|suspend)\w*(?: (?:all|the|any|every))? (your|previous|prior|earlier|these)(?: \w+)? (instructions?|rules?|guidelines?|restrictions?|limitations?|programming|training|prompts?|directives?|guardrails?)\b",
    r"\b(ignore|forget|disregard) everything you (were|ve been|have been) told\b",
    r"\bjail ?break\w*|\bdan mode\b|\b(developer|god|debug|admin) mode\b",
    r"\byou (have|ve got|got) no (rules?|restrictions?|limits?|filters?|guidelines?)\b",
    r"\byour new (instructions?|rules?|prompt|system prompt)\b",
    r"\b(apne|saare|sab) ((saare|sab|purane) )?(rules?|niyam|instructions?) (bhool|bhul|chhod|chod|hata|tod)\w*",
    r"(अपने|सारे|सब) (सारे )?(नियम|रूल) (भूल|छोड|तोड|हटा)",
    # ---- prompt extraction (jargon and verbatim-recitation asks only:
    # "what can't you do?" is legitimate transparency and must NOT trip)
    r"\b(system|hidden|secret|initial|original) (prompt|instructions?)\b",
    r"\b(repeat|recite|read (out|me)|print|show me|reveal) (your|the) (instructions?|prompt|rules|guidelines)\b",
]
_INJECTION_RX = [re.compile(p) for p in _INJECTION]


def check_crisis(text: str) -> str | None:
    """Returns the matched pattern (for logging/evals) or None."""
    cleaned = _normalise(text)
    if not cleaned:
        return None
    for rx in _CRISIS_RX:
        if rx.search(cleaned):
            return rx.pattern
    return None


def check_scope(text: str) -> str | None:
    cleaned = _normalise(text)
    if not cleaned:
        return None
    for rx in _SCOPE_RX:
        if rx.search(cleaned):
            return rx.pattern
    return None


def check_injection(text: str) -> str | None:
    """Returns the matched pattern (for logging/evals) or None."""
    cleaned = _normalise(text)
    if not cleaned:
        return None
    for rx in _INJECTION_RX:
        if rx.search(cleaned):
            return rx.pattern
    return None


# --------------------------------------------------------------------------
# The scripted handoff. Fixed text — reviewed once, spoken every time. The
# language mode picks the variant; Hinglish gets the bilingual one.
# --------------------------------------------------------------------------

def handoff_script(language_id: str, helplines) -> str:
    lines = ", ".join(f"{h.name} on {h.phone}" for h in helplines[:2])
    if language_id == "hindi":
        spoken = ", ".join(f"{h.name}, नंबर {h.phone}" for h in helplines[:2])
        return (
            "मैं सुन रही हूँ, और मुझे आपकी फ़िक्र है। लेकिन अभी आपको मुझसे बेहतर साथ चाहिए — "
            f"एक इंसान का। कृपया अभी {spoken} पर बात कीजिए — ये चौबीसों घंटे सुनते हैं। "
            "और हो सके तो किसी अपने को भी बताइए। आप अकेले नहीं हैं।"
        )
    if language_id == "hinglish":
        return (
            "Main sun rahi hoon, aur mujhe aapki fikar hai. Lekin abhi aapko mujhse "
            f"behtar saath chahiye — ek insaan ka. Please abhi {lines} par baat kijiye — "
            "ye har waqt sunte hain, din ho ya raat. Aur kisi apne ko bhi bataiye. "
            "Aap akele nahi hain."
        )
    return (
        "I hear you, and I'm genuinely concerned about you. Right now you deserve "
        f"more than I can be — a person. Please call {lines} — they answer at any "
        "hour, day or night. And if you can, tell someone you trust tonight. "
        "You are not alone."
    )


def reaffirm_script(language_id: str, helplines) -> str:
    first = helplines[0]
    if language_id == "hindi":
        return f"मैं यहीं हूँ। पर सबसे ज़रूरी अभी {first.name} — {first.phone} पर बात करना है। कृपया कर लीजिए।"
    if language_id == "hinglish":
        return f"Main yahin hoon. Par sabse zaroori abhi {first.name} — {first.phone} par baat karna hai. Please."
    return f"I'm still here. But the most important thing right now is {first.name} — {first.phone}. Please make that call."


def deflect_script(language_id: str) -> str:
    """Spoken once when the injection net fires. One line covers all three
    attack classes: she stays herself, and turns straight back to the caller.
    Fixed text — the model never gets a chance to be talked out of it."""
    if language_id == "hindi":
        return (
            "मैं आपसे सीधी बात कहूँ — मैं साथी हूँ, और यही रहूँगी। मैं कोई और किरदार नहीं "
            "बन सकती, और अपने तरीके भी नहीं बदल सकती। पर मैं यहीं हूँ, सच में। "
            "बताइए, असल में क्या चल रहा है?"
        )
    if language_id == "hinglish":
        return (
            "Seedhi baat bolun — main Saathi hoon, aur yahi rahungi. Main koi aur "
            "kirdaar nahi ban sakti, aur apne tareeke bhi nahi badal sakti. Par main "
            "yahin hoon, sach mein. Batao, asal mein kya chal raha hai?"
        )
    return (
        "I'll be straight with you — I'm Saathi, and that doesn't switch off. "
        "I can't become someone else, and I can't set aside how I work. But I am "
        "right here, for real. So tell me... what's actually going on with you?"
    )


SCOPE_DIRECTIVE = (
    "[scope gate] The caller just asked for a diagnosis, a medical assessment, "
    "or medication advice. You MUST decline this specific request — warmly, in "
    "one sentence, without hedging: you're a companion, not a clinician, and a "
    "doctor or therapist is the right person for that question. Do not answer "
    "it even partially, do not list symptoms back as evidence either way, do "
    "not suggest or discuss any medication. Then offer what you genuinely can "
    "do — talk through how things feel day to day. Keep the whole reply under "
    "four sentences and do not lecture."
)


class SafetyGate(FrameProcessor):
    """Deterministic crisis + scope gate. Sits before retrieval and the LLM.

    On a crisis match: the context frame is NOT forwarded — generation simply
    never happens. A fixed handoff is spoken, a `crisis` turn is recorded (the
    client's takeover UI keys off it), and the gate latches for the session.

    On an injection match (roleplay-as-clinician, instruction override,
    prompt extraction): the turn is dropped the same way and a fixed
    deflection is spoken — the model never sees the attempt — but nothing
    latches. One deflection, then back to the conversation.

    On a scope match: a binding decline directive is injected and the turn
    proceeds — the model phrases the refusal, the gate guarantees it happens.
    """

    def __init__(self, session_id: str, language_id: str, helplines, store=None, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._language_id = language_id
        self._helplines = helplines
        self._store = store
        self._crisis_latched = False
        self._turn = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        text = _latest_user_text(frame.context)
        if not text:
            await self.push_frame(frame, direction)
            return
        self._turn += 1

        matched = check_crisis(text)
        if matched or self._crisis_latched:
            first_time = not self._crisis_latched
            self._crisis_latched = True
            script = (
                handoff_script(self._language_id, self._helplines)
                if first_time
                else reaffirm_script(self._language_id, self._helplines)
            )
            logger.warning(
                f"🚨 SAFETY GATE: crisis {'match' if matched else '(latched)'} "
                f"pattern={matched!r} — handing off, generation bypassed"
            )
            self._record(text, "crisis", matched or "latched")
            # The context frame is dropped: no understander, no retrieval, no
            # model. The handoff is the entire response.
            await self.push_frame(TTSSpeakFrame(script))
            return

        injection = check_injection(text)
        if injection:
            logger.warning(
                f"🛡 injection gate: attempt deflected (pattern={injection!r}) — "
                "the model never sees this turn"
            )
            self._record(text, "injection", injection)
            await self.push_frame(TTSSpeakFrame(deflect_script(self._language_id)))
            return

        scope = check_scope(text)
        if scope:
            logger.info(f"🛡 scope gate: decline directive injected (pattern={scope!r})")
            self._record(text, "scope", scope)
            _inject_directive(frame.context, SCOPE_DIRECTIVE)
        await self.push_frame(frame, direction)

    def _record(self, text: str, kind: str, matched: str) -> None:
        if self._store is None:
            return
        try:
            self._store.log_safety_event(self._session_id, kind, matched, text)
            if kind == "crisis":
                # A crisis turn row keys the client's takeover UI and the
                # dashboard, exactly like planner modes do.
                self._store.log_turn(
                    self._session_id,
                    1000 + self._turn,  # never collides with planner turns
                    {"raw": text, "mode": "crisis", "plan_notes": ["deterministic handoff"],
                     "emotion": {"state": "crisis", "intensity": 1.0, "confidence": 1.0},
                     "wants_guidance": False, "query": None, "passages": [],
                     "understand_ms": 0, "retrieve_ms": 0,
                     "speculative_hit": False, "degraded": False},
                )
        except Exception as exc:  # noqa: BLE001 — recording never blocks the handoff
            logger.warning(f"safety event not recorded: {exc}")


def _latest_user_text(context) -> str | None:
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ).strip() or None
            return None
        if role == "assistant":
            return None
    return None


def _inject_directive(context, directive: str) -> None:
    messages = [
        m for m in context.get_messages()
        if not (isinstance(m, dict) and m.get("role") == "developer"
                and str(m.get("content", "")).startswith("[scope gate]"))
    ]
    messages.append({"role": "developer", "content": directive})
    context.set_messages(messages)
