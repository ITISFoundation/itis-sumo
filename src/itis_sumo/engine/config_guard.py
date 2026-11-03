"""Sanity checks for NIDR (Dakota input) configuration text."""

REQUIRED_BLOCKS = ("variables", "responses", "interface", "method")


def validate_nidr_config(conf: str) -> list[str]:
    """Cheap sanity-guard for a NIDR config string before handing it to the wheel.

    Does not parse NIDR (no grammar): only structural red flags that turn into
    cryptic Dakota abort codes. Returns a list of problems; empty means OK.
    """
    problems: list[str] = []
    if not conf or not conf.strip():
        problems.append("configuration is empty")
        return problems

    if conf.count("{") != conf.count("}"):
        problems.append(
            f"unbalanced braces: {conf.count('{')} '{{' vs {conf.count('}')} '}}'"
        )

    stripped = conf.strip().lower()
    for block in REQUIRED_BLOCKS:
        if not any(line.strip().startswith(block) for line in stripped.splitlines()):
            problems.append(f"missing required block: '{block}'")

    return problems
