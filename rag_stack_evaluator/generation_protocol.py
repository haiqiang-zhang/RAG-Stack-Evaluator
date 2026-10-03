"""Explicit generation semantics for current runs and historical quality replay.

The legacy protocol is limited to native static-quality evaluation. It dispatches
raw strings to ``LLM.generate`` and structured messages to ``LLM.chat`` exactly
as the original formal optimizer ablations did. It is never a serving protocol.
"""

CHAT_COMPLETIONS = "chat_completions"
LEGACY_COMPLETION = "legacy_completion"


def resolve_generation_protocol(params: dict, *, allow_legacy: bool = False) -> str:
    protocol = params.get("generation_protocol", CHAT_COMPLETIONS)
    if protocol not in (CHAT_COMPLETIONS, LEGACY_COMPLETION):
        raise ValueError(f"unsupported generation_protocol={protocol!r}")
    if protocol == LEGACY_COMPLETION:
        if not allow_legacy:
            raise ValueError(
                "generation_protocol=legacy_completion is restricted to native "
                "vllm static_gt quality evaluation; API and measured execution "
                "require chat_completions"
            )
        if "use_chat_template" in params and params["use_chat_template"] is not False:
            raise ValueError(
                "generation_protocol=legacy_completion requires omitting "
                "use_chat_template or setting it to false"
            )
    elif "use_chat_template" in params and params["use_chat_template"] is not True:
        raise ValueError(
            "local vLLM raw-completion mode is disabled; use_chat_template must "
            "be true unless generation_protocol=legacy_completion is explicit. "
            "Raw completions are unsupported by the default chat protocol."
        )
    return protocol


def reject_legacy_generation_protocol(config) -> None:
    """Reject historical quality settings before a measured deployment starts."""
    if isinstance(config, dict):
        if "generation_protocol" in config:
            resolve_generation_protocol(config)
        for value in config.values():
            reject_legacy_generation_protocol(value)
    elif isinstance(config, (list, tuple)):
        for value in config:
            reject_legacy_generation_protocol(value)
