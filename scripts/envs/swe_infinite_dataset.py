"""Prepare SWE Infinite SFT training dataset from the validator-mounted miner dataset.

Reads ``gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered`` (the only
whitelisted SFT dataset for this env) from the validator-mounted miner datasets
directory and rewrites every row's ``messages`` into the exact wire format the
live ``gradientsio/swe-infinite:v1`` evaluator (mini-swe-agent) actually uses.

Unlike the PvP game envs and intercode, swe_infinite is NOT tool-calling: the
live harness never sends a ``tools``/``tool_choice`` schema (see
docs/swe_infinite_eval_alignment.md §1, "Important"). The model must emit a
single fenced ```bash``` block per turn instead of an OpenAI function call, so
rows here are plain ``{"messages": [...]}`` with no ``tools`` key.

The dataset's own format diverges from the live eval format on several axes
that plain reformatting CAN fix (system prompt, user-turn XML wrapper,
tool-result wrapper, the submit command) and one it CANNOT (the issue prose
itself is LLM-paraphrased in the dataset vs. verbatim GitHub issue text at
eval time -- see docs/swe_infinite_eval_alignment.md §3.2's caveat). This
module implements exactly the deterministic §5 transform from that doc:
    - system message: hard-replaced with the live harness's 542-char prompt.
    - first user message: the dataset's structured "## Issue" block is
      reparsed (keeping only Title/Problem/Fix prose -- Root Cause/Relevant
      Interface/Repository Info are synthetic additions the live harness
      never shows) and re-wrapped as <pr_description>...</pr_description> +
      the static 4962-char <instructions> block.
    - assistant turns: the bare submit echo is rewritten to the compound
      ``&& git add -A && git diff --cached`` form the live harness actually
      requires to capture a patch (docs/swe_infinite_eval_alignment.md §3.4 --
      the dataset's own convention would never produce a capturable patch).
    - observation turns: "Observation: X" is rewrapped as
      "<returncode>0</returncode><output>X</output>" (returncode is not
      recoverable from the dataset -- a documented lossy step, not silently
      dropped).
    - correction-message turns pass through unchanged (already byte-identical
      to the live harness per §3.5).

Row filtering: only ``exit_status == "Submitted"`` rows are kept by default,
per the doc's own recommendation (§4's data-quality note) -- the other 96% of
rows never attempted a submission and would over-represent trajectories that
simply ran out of turn budget mid-exploration, without ever exercising the
one behavior (the submit rewrite above) this transform most needs to teach.

Dataset path convention (follows miner_dataset_loader.py):
    MINER_DATASETS_DIR/<hf_org>--<hf_repo>/   (-- replaces / in the HF repo name)
    MINER_DATASETS is a comma-separated list of those directory names.

Usage (run from /workspace/scripts/):
    python -m envs.swe_infinite_dataset --output_path /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from envs.miner_dataset_loader import _get_miner_datasets_inventory, _load_one_dataset

SWE_INFINITE_HF_REPO = "gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered"

# Only rows that actually reached a submission are kept by default -- see the
# module docstring's "Row filtering" note.
_KEEP_EXIT_STATUSES = frozenset({"Submitted"})


# ---------------------------------------------------------------------------
# Live wire format -- verbatim from docs/swe_infinite_eval_alignment.md §2.1/§2.2
# (empirically captured from the real gradientsio/swe-infinite:v1 image; see
# that doc's "Methodology" section for how to re-capture if the image updates).
# ---------------------------------------------------------------------------

LIVE_SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

LIVE_INSTRUCTIONS_BLOCK = """<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements described above.
Your task is to make changes to source files in the current directory to resolve the described issue in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

For each response:
1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries
- MODIFY: Regular source code files in /app (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)
- NEVER add or modify unit tests. Your job is ONLY to implement or fix the source code.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a simple script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Command Execution Rules
You are operating in an environment where
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like this:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

Commands must be specified in a single bash code block:

```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example_response>

Example of an INCORRECT response:
<example_response>
THOUGHT: I need to examine the codebase and then look at a specific file. I'll run multiple commands to do this.

```bash
ls -la
```

Now I'll read the file:

```bash
cat file.txt
```
</example_response>

If you need to run multiple commands, either:
1. Combine them in one block using && or ||
```bash
command1 && command2 || echo "Error occurred"
```

2. Wait for the first command to complete, see its output, then issue the next command in your following response.

## Environment Details
- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- If a command isn't available, you can install it

## Useful Command Examples

### Create a new file:
```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:
```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run
```bash
anything
```

## Submission
When you've completed your work (reading, editing, testing), and cannot make further progress
issue exactly the following command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```

This command will submit your work.
You cannot continue working (reading, editing, testing) in any way on this task after submitting.
</instructions>"""


# ---------------------------------------------------------------------------
# Dataset lookup
# ---------------------------------------------------------------------------

def _find_swe_infinite_dataset() -> "tuple[str, Path] | None":
    """Return (hf_name, local_path) for the SWE Infinite dataset, or None."""
    for hf_name, local in _get_miner_datasets_inventory():
        if "swe-zero" in hf_name.lower():
            return hf_name, local
    return None


# ---------------------------------------------------------------------------
# Dataset's first-user-turn parsing -- see docs/swe_infinite_eval_alignment.md §3.2
# ---------------------------------------------------------------------------

# Matches the dataset's own template (§3.2):
#   Please solve this issue in the repository {repo}.
#
#   ## Issue
#
#   **Title**
#   {title}
#
#   **Problem**
#   {problem}
#
#   **Root Cause**
#   {root_cause}
#   ... (Fix / Expected Behavior, Risk & Validation, ## Relevant Interface, ## Repository Info)
#
# Only Title/Problem/Fix are kept -- the rest are synthetic additions the live
# harness never shows the model (see module docstring).
_RE_FIELD = {
    "title":   re.compile(r"\*\*Title\*\*\n(.*?)(?=\n\*\*|\n##|\Z)", re.S),
    "problem": re.compile(r"\*\*Problem\*\*\n(.*?)(?=\n\*\*|\n##|\Z)", re.S),
    "fix":     re.compile(r"\*\*Fix / Expected Behavior\*\*\n(.*?)(?=\n\*\*|\n##|\Z)", re.S),
}


def _extract_issue_prose(dataset_user_content: str) -> str:
    """Reconstruct issue prose from the dataset's structured '## Issue' block.

    Keeps only Title/Problem/Fix (the live eval's <pr_description> is close to
    a raw issue body, not a Root-Cause/Relevant-Interface/Repository-Info
    structured writeup -- see docs/swe_infinite_eval_alignment.md §3.2/§5).
    Falls back to the raw content if the expected fields aren't found (e.g. a
    row that doesn't match the documented template), so a schema drift fails
    soft (skipped by the score/shape checks downstream) rather than crashing.
    """
    parts = []
    for key in ("title", "problem", "fix"):
        m = _RE_FIELD[key].search(dataset_user_content)
        if m:
            parts.append(m.group(1).strip())
    if not parts:
        return dataset_user_content.strip()
    return "\n\n".join(parts)


def _extract_repo(row: dict[str, Any], dataset_user_content: str) -> str:
    if row.get("repo"):
        return str(row["repo"])
    m = re.search(r"Please solve this issue in the repository (\S+?)\.", dataset_user_content)
    return m.group(1) if m else "unknown/unknown"


def _build_pr_description_turn(row: dict[str, Any], dataset_user_content: str) -> dict[str, Any]:
    """Mirrors docs/swe_infinite_eval_alignment.md §5's transform_row step 2.

    Language is deliberately omitted: the dataset's own value is always the
    literal string "unknown" (confirmed in the doc across a 5,000-row sample)
    and there's no reliable offline way to re-derive the real language from
    this dataset's fields alone -- omitting the line matches the doc's own
    suggested fallback rather than propagating a value that never appears in
    real eval traffic.
    """
    repo = _extract_repo(row, dataset_user_content)
    issue_body = _extract_issue_prose(dataset_user_content)
    pr_description = f"Consider the following issue or PR description:\nRepository: {repo}\n\n{issue_body}"
    content = f"<pr_description>\n{pr_description}\n</pr_description>\n\n{LIVE_INSTRUCTIONS_BLOCK}"
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Remaining-turn rewrites -- docs/swe_infinite_eval_alignment.md §5
# ---------------------------------------------------------------------------

# Bare submit -> compound submit (the fix for the "never produces a
# capturable patch" mismatch, §3.4).
_RE_BARE_SUBMIT = re.compile(
    r"```bash\s*\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*\n```"
)
_COMPOUND_SUBMIT = (
    "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"
)


def _rewrite_assistant_turn(m: dict[str, Any]) -> dict[str, Any]:
    content = _RE_BARE_SUBMIT.sub(_COMPOUND_SUBMIT, m.get("content") or "")
    return {"role": "assistant", "content": content}


# Dataset's own truncation convention (§3.3), distinct from the live
# harness's <warning>/<output_head>/<elided_chars>/<output_tail> wrapper.
_RE_TRUNCATED = re.compile(
    r"^(?P<head>.*?)\n\n\.\.\. \(output truncated, (?P<n>\d+) chars elided\) \.\.\.\n\n(?P<tail>.*)$",
    re.S,
)

# Returncode is not recorded by the dataset at all -- defaulting to "0" is a
# documented lossy step (docs/swe_infinite_eval_alignment.md §5 "Known-lossy
# steps" #1), not a silently-invented value.
_UNRECOVERABLE_RETURNCODE = "0"


def _rewrite_observation_turn(m: dict[str, Any]) -> dict[str, Any]:
    content = m.get("content") or ""
    if not content.startswith("Observation:"):
        # Correction message (§2.6/§3.5) or an already-special turn -- pass
        # through unchanged, it's already byte-identical to the live harness.
        return m

    body = content[len("Observation:"):].lstrip("\n")
    if body.startswith(" "):
        body = body[1:]  # dataset uses "Observation: X" (space, no newline) for short outputs

    trunc = _RE_TRUNCATED.match(body)
    if trunc:
        wrapped = (
            f"<returncode>{_UNRECOVERABLE_RETURNCODE}</returncode>\n"
            "<warning>\n"
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
            "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
            "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
            "</warning><output_head>\n"
            f"{trunc['head']}\n"
            "<elided_chars>\n"
            f"{trunc['n']} characters elided\n"
            "</elided_chars>\n"
            "<output_tail>\n"
            f"{trunc['tail']}\n"
            "</output_tail>"
        )
    else:
        wrapped = f"<returncode>{_UNRECOVERABLE_RETURNCODE}</returncode>\n<output>\n{body}\n</output>"

    return {"role": "user", "content": wrapped}


# ---------------------------------------------------------------------------
# Trailing-window trim -- caps how much of a long trajectory's MIDDLE gets
# kept, so the TAIL (ending at the submit turn) always survives.
#
# swe_infinite rows are the only full growing multi-turn conversations this
# SFT pipeline produces (every other env's rows are single flattened turns,
# see docs/SFT_ALIGNMENT_PLAN.md §3), but max_length is a fixed 4096 tokens
# applied uniformly across every env (sft_env_config.py::get_run_cmd). A
# generic head-first truncation (train_sft_env.py::tokenize_and_mask slices
# ids[:max_length]) keeps the BEGINNING of an overlong trajectory and cuts
# the END -- exactly where the submit turn lives, the one behavior this
# dataset most needs to teach (the bare->compound submit rewrite above; rows
# are already filtered to exit_status=="Submitted" specifically for this).
# Trimming here instead keeps the anchor (system + pr_description -- the task
# itself) plus only the trailing messages, so the kept window's tail always
# matches the trajectory's actual tail regardless of total length.
#
# Known lossy step (same spirit as the "Known-lossy steps" in
# docs/swe_infinite_eval_alignment.md §5): a trimmed window may reference
# files/decisions established only in the dropped middle turns. Message-count
# based (not token-based) since dataset generation doesn't depend on which
# tokenizer will train on it, matching generate_trajectories.py's own
# _sliding_windows precedent for the (currently unused) non-flat game-env
# path. 6 trailing messages = 3 (assistant, observation) round trips right
# before the end; tune if real-dataset token stats say otherwise.
# ---------------------------------------------------------------------------

_MAX_TRAILING_MESSAGES = 6


def _window_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep [system, pr_description] + at most the last _MAX_TRAILING_MESSAGES.

    No-op for trajectories already shorter than the cap.
    """
    if len(messages) <= 2:
        return messages
    anchor, rest = messages[:2], messages[2:]
    if len(rest) <= _MAX_TRAILING_MESSAGES:
        return messages
    return anchor + rest[-_MAX_TRAILING_MESSAGES:]


# ---------------------------------------------------------------------------
# Row transform
# ---------------------------------------------------------------------------

def transform_row(row: dict[str, Any]) -> "dict[str, Any] | None":
    """Rewrite one raw dataset row's messages into the live wire format.

    Returns None if the row doesn't match the expected shape (missing
    messages, wrong leading roles) rather than raising -- schema drift in a
    single row should be skipped, not abort the whole dataset build.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return None

    out: list[dict[str, Any]] = [{"role": "system", "content": LIVE_SYSTEM_PROMPT}]
    out.append(_build_pr_description_turn(row, messages[1].get("content") or ""))

    for m in messages[2:]:
        if not isinstance(m, dict) or "role" not in m:
            continue
        if m["role"] == "assistant":
            out.append(_rewrite_assistant_turn(m))
        else:
            out.append(_rewrite_observation_turn(m))

    return {"messages": _window_trajectory(out)}


def _row_to_example(row: dict[str, Any]) -> "dict[str, Any] | None":
    if row.get("exit_status") not in _KEEP_EXIT_STATUSES:
        return None
    example = transform_row(row)
    if example is None:
        return None
    # JSON-encoded: see docs/MULTI_ENV_DATASET_MERGE_BUG.md -- this env's
    # `messages` column (plain {role, content}, no tool_calls/tools at all)
    # is a strict subset of the tool-calling envs' shape, which
    # concatenate_datasets refuses to align as a native List(struct) column
    # when generate_trajectories.py's multi-env path merges per-env datasets.
    # Storing as a string sidesteps Arrow schema alignment entirely; decoded
    # back in train_sft_env.py::tokenize_and_mask. transform_row() itself is
    # left returning the native structure so it stays directly unit-testable.
    return {"messages": json.dumps(example["messages"])}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_swe_infinite_sft_dataset() -> "DatasetDict | None":
    """Find and convert the SWE Infinite miner dataset; return DatasetDict or None."""
    found = _find_swe_infinite_dataset()
    if found is None:
        print(
            f"[swe_infinite_dataset] '{SWE_INFINITE_HF_REPO}' not found in miner dataset inventory.\n"
            "  Make sure MINER_DATASETS includes "
            "'gradients-io-tournaments--SWE-ZERO-12M-trajectories-filtered' "
            "and MINER_DATASETS_DIR is set.",
            flush=True,
        )
        return None

    hf_name, local = found
    print(f"[swe_infinite_dataset] Loading {hf_name} from {local}", flush=True)

    raw = _load_one_dataset(local)
    if raw is None:
        print(f"[swe_infinite_dataset] Failed to load dataset from {local}", flush=True)
        return None

    print(f"[swe_infinite_dataset] Loaded {len(raw)} rows", flush=True)

    all_examples: list[dict[str, Any]] = []
    skipped = 0
    for row in raw:
        example = _row_to_example(dict(row))
        if example is not None:
            all_examples.append(example)
        else:
            skipped += 1

    print(
        f"[swe_infinite_dataset] Converted to {len(all_examples)} trajectories "
        f"({skipped}/{len(raw)} rows skipped -- not exit_status=Submitted or malformed)",
        flush=True,
    )

    if not all_examples:
        return None

    return DatasetDict({"train": Dataset.from_list(all_examples)})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert the SWE Infinite whitelisted dataset to SFT training format."
    )
    p.add_argument("--output_path", required=True, help="Where to save the DatasetDict")
    args = p.parse_args()

    dd = build_swe_infinite_sft_dataset()
    if dd is None:
        sys.exit(1)

    dd.save_to_disk(args.output_path)
    print(
        f"[swe_infinite_dataset] Saved {len(dd['train'])} examples → {args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
