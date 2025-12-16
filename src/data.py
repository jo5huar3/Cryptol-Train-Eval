from copy import deepcopy
from datasets import Dataset, load_dataset
from typing import List, Dict, Any

SKIPPED_FILES = set()

def _split_assistant_into_chunks(
    prefix_msgs,
    assistant_msg,
    tokenizer,
    max_len: int,
):
    """
    Return a list of new conversations of the form:

        [*prefix_msgs, {"role": "assistant", "content": part_i}]

    where each conversation is <= max_len tokens.
    """
    assert assistant_msg["role"] == "assistant"

    text = assistant_msg["content"]
    lines = text.splitlines(keepends=True)

    chunks = []
    cur_lines: list[str] = []

    def conv_tokens(lines_fragment):
        conv = prefix_msgs + [{
            "role": "assistant",
            "content": "".join(lines_fragment),
        }]
        ids = tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
        )
        return len(ids)

    for line in lines:
        tentative = cur_lines + [line]
        if conv_tokens(tentative) <= max_len:
            cur_lines.append(line)
        else:
            if cur_lines:
                # flush current chunk
                chunks.append(prefix_msgs + [{
                    "role": "assistant",
                    "content": "".join(cur_lines),
                }])
                cur_lines = [line]
            else:
                # single line is too long (rare) – you could fall back to
                # character-based splitting here if you ever hit this case
                chunks.append(prefix_msgs + [{
                    "role": "assistant",
                    "content": line,
                }])
                cur_lines = []

    if cur_lines:
        chunks.append(prefix_msgs + [{
            "role": "assistant",
            "content": "".join(cur_lines),
        }])

    return chunks


def _chunk_messages_by_tokens(
    messages,
    tokenizer,
    max_len: int,
    source_name: str | None = None,
    long_msg_mode: str = "skip",
):
    chunks = []
    current = []

    def tokens_for(msgs):
        # normalize first (stringify, etc.)
        msgs = _normalize_messages(msgs)

        # IMPORTANT: never call chat_template on assistant-first sequences
        if msgs and msgs[0]["role"] == "assistant":
            msgs = [{"role": "user", "content": ""}] + msgs

        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
        )
        return len(ids)


    def log_skip(n_tokens: int):
        msg = f"Skipping over-long single message with {n_tokens} tokens"
        if source_name is not None:
            msg += f" from {source_name}"
            SKIPPED_FILES.add(source_name)
        print(msg)

    for msg in messages:
        tentative = current + [msg]
        length = tokens_for(tentative)

        if length <= max_len:
            current = tentative
            continue

        # At this point, adding `msg` made us too long.
        single_len = tokens_for([msg])

        # Case 1: message itself fits; just close current chunk.
        if single_len <= max_len:
            if current:
                chunks.append(current)
            current = [msg]
            continue

        # Case 2: SINGLE MESSAGE is itself too long
        if (
            long_msg_mode == "split"
            and msg.get("role") == "assistant"
        ):
            # We *don't* want a chunk that is just `current` with no assistant,
            # so we *don't* append `current` alone.
            prefix = current

            # Split the assistant message into several smaller convs
            split_convs = _split_assistant_into_chunks(
                prefix_msgs=prefix,
                assistant_msg=msg,
                tokenizer=tokenizer,
                max_len=max_len,
            )

            chunks.extend(split_convs)
            current = []  # reset
        else:
            # Old behavior: skip
            if current:
                chunks.append(current)
            log_skip(single_len)
            current = []

    if current:
        chunks.append(current)

    return chunks

def explode_long_conversations(
    raw_ds: Dataset,
    tokenizer,
    max_len: int,
    long_msg_mode: str = "skip",   # "skip" (current behavior) or "split"
) -> Dataset:
    """
    Take the original `raw` Dataset with a `messages` column and return
    a new Dataset where long conversations have been split into multiple
    rows of <= max_len tokens.

    All other columns (filetype, filename, etc.) are copied through.
    Two extra columns are added:
      - orig_conv_idx: the original row index in `raw`
      - chunk_idx: which chunk number (0, 1, 2, ...) from that conversation

    long_msg_mode:
      - "skip": if a *single* message > max_len, skip it (old behavior)
      - "split": if an *assistant* message > max_len, split it into multiple
                 assistant replies, each in its own conversation
    """
    assert long_msg_mode in {"skip", "split"}

    new_rows = []

    for i in range(len(raw_ds)):
        row = raw_ds[i]
        msgs = row["messages"]

        # normalize messages if you have this helper
        msgs_norm = _normalize_messages(msgs)

        # use whatever your filename column is called; adjust if needed
        source_name = row.get("filename", f"row_{i}")

        chunks = _chunk_messages_by_tokens(
            msgs_norm,
            tokenizer,
            max_len,
            source_name=str(source_name),
            long_msg_mode=long_msg_mode,   # <-- pass mode through
        )

        for j, chunk in enumerate(chunks):
            new_row = dict(row)
            new_row["messages"] = chunk
            new_row["orig_conv_idx"] = i
            new_row["chunk_idx"] = j
            new_rows.append(new_row)

    print(f"Expanded {len(raw_ds)} original rows into {len(new_rows)} chunks")

    # Optional: summary of which files had over-long messages
    if SKIPPED_FILES:
        print("\nFiles with at least one skipped over-long message:")
        for name in sorted(SKIPPED_FILES):
            print("  ", name)

    return Dataset.from_list(new_rows)

def _normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize messages into strict chat-template friendly form."""
    def to_text(content):
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)

    sys_parts = []
    seq = []

    for m in msgs:
        role = (m.get("role") or "").strip()
        content = to_text(m.get("content", ""))

        if not role:
            continue

        # If your data has tool/function roles, map or drop them.
        if role in {"tool", "function", "observation"}:
            role = "assistant"

        if role == "system":
            if content.strip():
                sys_parts.append(content)
            continue

        if not content.strip():
            continue

        # merge consecutive same-role turns
        if seq and seq[-1]["role"] == role:
            seq[-1]["content"] += "\n\n" + content
        else:
            seq.append({"role": role, "content": content})

    # drop leading assistants (template wants user first after optional system)
    while seq and seq[0]["role"] == "assistant":
        seq.pop(0)

    # place system at top (single message)
    if sys_parts:
        sys_text = "\n\n".join(sys_parts)
        seq = [{"role": "system", "content": sys_text}] + seq

    return seq


def load_or_make_dataset(jsonl_path: str):
    if jsonl_path:
        if jsonl_path.endswith(".jsonl"):
            ds = load_dataset("json", data_files=jsonl_path, split="train")
        else:
            # assume JSON array
            ds = load_dataset("json", data_files=jsonl_path, split="train")
        # basic validation
        assert "messages" in ds.column_names, "Dataset must have a 'messages' column."
        return ds
    else:
        # Toy dataset (few-shot) for a quick smoke test
        toy = [
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about the moon."},
                {"role": "assistant", "content": "Silent silver orb\nDrifting high in velvet night\nDreams glow in cool light."}
            ]},
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 17 * 12?"},
                {"role": "assistant", "content": "17 * 12 = 204."}
            ]},
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give me a short Python function that returns the square of a number."},
                {"role": "assistant", "content": "def square(x):\n    return x * x"}
            ]},
        ]
        return Dataset.from_list(toy)