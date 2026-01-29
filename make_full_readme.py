from pathlib import Path
import re

ROOT = Path(".").resolve()
OUTPUT = ROOT / "README.md"

IMG_SRC_RE = re.compile(r'src="([^":]+)"')

MD_HEADING_RE = re.compile(r'^(#{1,6})(\s+.*)$')

def rewrite_paths(text: str, readme_dir: Path) -> str:
    def repl(match):
        src = match.group(1)

        # Ignore absolute URLs
        if src.startswith(("http://", "https://", "/")):
            return match.group(0)

        # Compute new path relative to ROOT
        abs_path = (readme_dir / src).resolve()
        new_src = abs_path.relative_to(ROOT).as_posix()

        return f'src="{new_src}"'

    return IMG_SRC_RE.sub(repl, text)


def strip_initial_h1(text: str) -> str:
    lines = text.splitlines()

    # Find first non-empty line
    first_nonempty = None
    for i, line in enumerate(lines):
        if line.strip():
            first_nonempty = i
            break

    if first_nonempty is None:
        return ""

    # If the file starts with a single top-level title, drop it when embedding.
    if lines[first_nonempty].lstrip().startswith("# "):
        del lines[first_nonempty]

    # Trim leading blank lines after removal
    while lines and not lines[0].strip():
        lines.pop(0)

    return "\n".join(lines)


def demote_markdown_headings(text: str, by: int = 1) -> str:
    if by <= 0:
        return text

    out_lines: list[str] = []
    in_fence = False

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue

        if in_fence:
            out_lines.append(line)
            continue

        m = MD_HEADING_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        hashes, rest = m.group(1), m.group(2)
        new_level = min(6, len(hashes) + by)
        out_lines.append("#" * new_level + rest)

    return "\n".join(out_lines)


readmes = sorted(
    p for p in ROOT.rglob("README.md")
    if p.parent != ROOT
)

with OUTPUT.open("w", encoding="utf-8") as out:
    # Include introduction.md if it exists
    intro_path = ROOT / "introduction.md"
    if intro_path.exists():
        content = intro_path.read_text(encoding="utf-8")
        out.write(content.strip())
        out.write("\n\n---\n")
    
    for readme in readmes:
        rel_path = readme.parent.relative_to(ROOT)

        out.write(f"\n\n## {rel_path}\n\n")

        content = readme.read_text(encoding="utf-8")
        content = rewrite_paths(content, readme.parent)
        content = strip_initial_h1(content)
        content = demote_markdown_headings(content, by=1)

        out.write(content.strip())
        out.write("\n")

print(f"Combined {len(readmes)} README files into {OUTPUT}")
