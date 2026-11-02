#!/usr/bin/env python3
"""Keep restricted data out of Claude's context (PreToolUse).

Denies any file tool (Read, Edit, Write, Grep, Glob, NotebookEdit, ...) whose
input names a restricted path, and any Bash command that names one, except:

    python3 .claude/skills/synthetic-data/generate.py <args>   (runs ON the data)
    ls <restricted dir>                                        (file names only)

Both exceptions must be a single plain command: no ; | & $ ` < > ( ) or newline,
so a read cannot be chained onto them.

Restricted = any path with a `restricted/` or `confidential/` segment (the
directories `.claude/rules/confidential-data.md` already names). Add your own
with CLAUDE_RESTRICTED_DIRS=/abs/path:/other/path.

LIMITS, stated plainly: a command that reads files without naming them
(`grep -r . `, a script that globs) is not caught. The `Read(...)` deny rules in
settings.json back this up; neither is a proof. Fails CLOSED: if the input
cannot be parsed and mentions a restricted word, the call is denied.
"""
import json
import os
import re
import sys

SEGMENT = re.compile(r"(^|[/\\\s'\"=])(restricted|confidential)([/\\]|$|[\s'\"])", re.I)
EXTRA = [p for p in os.environ.get("CLAUDE_RESTRICTED_DIRS", "").split(":") if p]
PLAIN = r"[^;|&$`<>()\n]*"
ALLOWED_BASH = [
    re.compile(rf"^\s*python3?\s+(\S*/)?\.claude/skills/synthetic-data/generate\.py\b{PLAIN}$"),
    re.compile(rf"^\s*ls\b{PLAIN}$"),
]


def restricted(text):
    return bool(SEGMENT.search(text)) or any(p in text for p in EXTRA)


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": reason}}))
    sys.exit(0)


def main():
    raw = sys.stdin.read()
    try:
        event = json.loads(raw)
        tool, args = event.get("tool_name", ""), event.get("tool_input", {}) or {}
    except Exception:
        if restricted(raw):
            deny("restricted-data-guard: unparseable tool call mentioning restricted data (fail closed).")
        return
    if tool == "Bash":
        cmd = args.get("command", "")
        if restricted(cmd) and not any(p.match(cmd) for p in ALLOWED_BASH):
            deny("restricted-data-guard: Claude must not read restricted data. "
                 "Run /synthetic-data to build a synthetic copy and work on data/synthetic/ instead.")
        return
    # path-like fields only: file *content* may mention restricted dirs (docs, this skill)
    if restricted(" ".join(str(args.get(k, "")) for k in ("file_path", "path", "notebook_path", "pattern", "glob"))):
        deny(f"restricted-data-guard: {tool} on a restricted path is blocked. "
             "Work on the synthetic copy in data/synthetic/ (see /synthetic-data).")


if __name__ == "__main__":
    main()
