#!/usr/bin/env python3
"""Fix malformed //...); lines and replace *_v template calls with raw intrinsics."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def fix_comment_swallowed_close(stripped: str) -> str | None:
    """If line ends with ); inside // comment only, split closing paren to its own line."""
    if "//" not in stripped or not stripped.rstrip().endswith(");"):
        return None
    idx = stripped.rfind("//")
    code_before = stripped[:idx]
    if ");" in code_before.rstrip():
        return None
    without_close = stripped[:-2]
    indent = len(stripped) - len(stripped.lstrip(" "))
    close_indent = max(0, indent - 4)
    return without_close + "\n" + " " * close_indent + ");"


def split_top_level_args(inner: str) -> list[str]:
    """Split on commas at depth 0; skip `// ...` line comments so commas inside comments do not split."""
    args: list[str] = []
    cur: list[str] = []
    depth = 0
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if depth == 0 and ch == "/" and i + 1 < n and inner[i + 1] == "/":
            while i < n and inner[i] != "\n":
                i += 1
            continue
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(cur).strip()
            if piece:
                args.append(piece)
            cur = []
        else:
            cur.append(ch)
        i += 1
    piece = "".join(cur).strip()
    if piece:
        args.append(piece)
    return args


def strip_trailing_comment(s: str) -> str:
    return s.split("//", 1)[0].strip()


def find_balanced_call(text: str, start: int, open_paren: int) -> tuple[int, int] | None:
    """Return (open_paren, end_exclusive) where end is index after closing ')'."""
    depth = 0
    p = open_paren
    while p < len(text):
        c = text[p]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return open_paren, p + 1
        p += 1
    return None


def replace_tagged_calls(text: str, tag: str, replacer) -> str:
    out: list[str] = []
    i = 0
    while True:
        j = text.find(tag, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        open_paren = j + len(tag)
        if open_paren >= len(text) or text[open_paren] != "(":
            raise RuntimeError(f"Expected '(' after {tag} at {j}")
        span = find_balanced_call(text, j, open_paren)
        if span is None:
            raise RuntimeError(f"Unbalanced call for {tag} at {j}")
        _, end = span
        whole = text[j:end]
        out.append(replacer(whole))
        i = end
    return "".join(out)


def repl_add_like(intrinsic: str, whole: str) -> str:
    # whole = tag + "(" + inner + ")"
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 10:
        raise RuntimeError(f"{intrinsic}: expected 10 args, got {len(args)} in {whole[:120]}...")
    dst, s0, s1 = args[0], args[1], args[2]
    nums = [strip_trailing_comment(a) for a in args[3:]]
    return (
        f"{intrinsic}(Ub<float>({dst}), Ub<float>({s0}), Ub<float>({s1}),\n"
        f"                {nums[0]},\n"
        f"                {nums[1]},\n"
        f"                {nums[2]},\n"
        f"                {nums[3]},\n"
        f"                {nums[4]},\n"
        f"                {nums[5]},\n"
        f"                {nums[6]}\n"
        f"            )"
    )


def repl_cadd(whole: str) -> str:
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 6:
        raise RuntimeError(f"vcadd: expected 6 args, got {len(args)}")
    dst, src = args[0], args[1]
    nums = [strip_trailing_comment(a) for a in args[2:]]
    return (
        f"vcadd(Ub<float>({dst}), Ub<float>({src}),\n"
        f"                {nums[0]},\n"
        f"                {nums[1]},\n"
        f"                {nums[2]},\n"
        f"                {nums[3]},\n"
        f"                0\n"
        f"            )"
    )


def repl_unary_vec(intrinsic: str, whole: str) -> str:
    """exp_v / ln_v: dst, src, repeat + 4 stride params (5 numeric tail, 7 args total)."""
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 7:
        raise RuntimeError(f"{intrinsic}: expected 7 args, got {len(args)}")
    dst, src = args[0], args[1]
    nums = [strip_trailing_comment(a) for a in args[2:]]
    return (
        f"{intrinsic}(Ub<float>({dst}), Ub<float>({src}),\n"
        f"                {nums[0]},\n"
        f"                {nums[1]},\n"
        f"                {nums[2]},\n"
        f"                {nums[3]},\n"
        f"                {nums[4]}\n"
        f"            )"
    )


def repl_scalar_bin(intrinsic: str, whole: str) -> str:
    """adds_v / muls_v: dst, src, scalar, repeat + 4 strides."""
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 8:
        raise RuntimeError(f"{intrinsic}: expected 8 args, got {len(args)}")
    dst, src, scalar = args[0], args[1], args[2]
    nums = [strip_trailing_comment(a) for a in args[3:]]
    return (
        f"{intrinsic}(Ub<float>({dst}), Ub<float>({src}), {scalar},\n"
        f"                {nums[0]},\n"
        f"                {nums[1]},\n"
        f"                {nums[2]},\n"
        f"                {nums[3]},\n"
        f"                {nums[4]}\n"
        f"            )"
    )


def repl_brcb_uint32(whole: str) -> str:
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 5:
        raise RuntimeError(f"vbrcb: expected 5 args, got {len(args)}")
    dst, src = args[0], args[1]
    nums = [strip_trailing_comment(a) for a in args[2:]]
    return (
        "vbrcb((__ubuf__ uint32_t*)Ub("
        + dst
        + "), (__ubuf__ uint32_t*)Ub("
        + src
        + "),\n"
        f"            {nums[0]},\n"
        f"            {nums[1]},\n"
        f"            {nums[2]}\n"
        f"        )"
    )


def repl_cmax(whole: str) -> str:
    open_idx = whole.index("(")
    inner = whole[open_idx + 1 : whole.rfind(")")]
    args = split_top_level_args(inner)
    if len(args) != 6:
        raise RuntimeError(f"vcmax: expected 6 args, got {len(args)}")
    dst, src = args[0], args[1]
    nums = [strip_trailing_comment(a) for a in args[2:]]
    return (
        f"vcmax(Ub<float>({dst}), Ub<float>({src}),\n"
        f"                {nums[0]},\n"
        f"                {nums[1]},\n"
        f"                {nums[2]},\n"
        f"                {nums[3]},\n"
        f"                static_cast<Order_t>(ORDER_ONLY_VALUE)\n"
        f"            )"
    )


def process_file(path: Path) -> None:
    raw = path.read_text()
    lines = raw.splitlines(keepends=True)
    fixed_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n\r")
        ends_nl = line.endswith("\n")
        rep = fix_comment_swallowed_close(stripped)
        if rep is not None:
            fixed_lines.append(rep + ("\n" if ends_nl else ""))
        else:
            fixed_lines.append(line)
    text = "".join(fixed_lines)

    # Longer tags first so shorter names do not match inside longer template names.
    text = replace_tagged_calls(text, "cadd_v<ArchType::ASCEND_V220, float>", repl_cadd)
    text = replace_tagged_calls(
        text, "cmax_v<ArchType::ASCEND_V220, float, ORDER_ONLY_VALUE>", repl_cmax
    )
    text = replace_tagged_calls(text, "muls_v<ArchType::ASCEND_V220, float>", lambda w: repl_scalar_bin("vmuls", w))
    text = replace_tagged_calls(text, "adds_v<ArchType::ASCEND_V220, float>", lambda w: repl_scalar_bin("vadds", w))
    text = replace_tagged_calls(text, "brcb_v<ArchType::ASCEND_V220, uint32_t>", repl_brcb_uint32)
    text = replace_tagged_calls(text, "add_v<ArchType::ASCEND_V220, float>", lambda w: repl_add_like("vadd", w))
    text = replace_tagged_calls(text, "sub_v<ArchType::ASCEND_V220, float>", lambda w: repl_add_like("vsub", w))
    text = replace_tagged_calls(text, "div_v<ArchType::ASCEND_V220, float>", lambda w: repl_add_like("vdiv", w))
    text = replace_tagged_calls(text, "max_v<ArchType::ASCEND_V220, float>", lambda w: repl_add_like("vmax", w))
    text = replace_tagged_calls(text, "mul_v<ArchType::ASCEND_V220, float>", lambda w: repl_add_like("vmul", w))
    text = replace_tagged_calls(text, "exp_v<ArchType::ASCEND_V220, float>", lambda w: repl_unary_vec("vexp", w))
    text = replace_tagged_calls(text, "ln_v<ArchType::ASCEND_V220, float>", lambda w: repl_unary_vec("vln", w))

    path.write_text(text)


def main() -> None:
    p = Path(sys.argv[1])
    process_file(p)


if __name__ == "__main__":
    main()
