"""CHANGELOG.md section extraction (standard library only)."""
import re


def section(changelog_text: str, version: str) -> str | None:
    """Extract the body of one release section from a changelog.

    :param changelog_text: full text of ``CHANGELOG.md``.
    :param version: version to look up, e.g. ``"1.1.0"``.
    :return: the stripped text between ``## [<version>] - YYYY-MM-DD`` and the next
        ``## [`` heading (or end of text), or ``None`` if the section is absent.
    """
    heading = re.search(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}[ \t]*$", changelog_text, re.M
    )
    if heading is None:
        return None
    rest = changelog_text[heading.end():]
    nxt = re.search(r"^## \[", rest, re.M)
    return (rest[: nxt.start()] if nxt else rest).strip()
