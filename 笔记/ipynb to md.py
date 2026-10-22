#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipynb2md.py —— 把 Jupyter Notebook (.ipynb) 批量转成 Markdown

用法:
    # 转换当前文件夹下所有 ipynb
    python "ipynb to md.py"

    # 转换指定文件夹下所有 ipynb
    python "ipynb to md.py" --dir ./notebooks

    # 转换指定文件（一个或多个）
    python "ipynb to md.py" "day 1 可视化.ipynb" "day 2 集合与逻辑.ipynb"

    # 不包含执行结果（推荐，md 更干净）
    python "ipynb to md.py" --no-output

    # 图片单独保存
    python "ipynb to md.py" --image-dir images
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def join_source(src) -> str:
    if src is None:
        return ""
    if isinstance(src, list):
        return "".join(src)
    return src


class NotebookConverter:
    def __init__(self, out_path: Path, include_output=True, image_dir=None,
                 image_prefix="image", include_html=False):
        self.out_path = out_path
        self.include_output = include_output
        self.image_dir = image_dir
        self.image_prefix = image_prefix
        self.include_html = include_html
        self._img_idx = 0

    def _rel(self, path: Path) -> str:
        rel = os.path.relpath(path, self.out_path.parent)
        return rel.replace(os.sep, "/")

    def _save_binary_image(self, b64: str, ext: str) -> str:
        self._img_idx += 1
        path = self.image_dir / f"{self.image_prefix}_{self._img_idx}.{ext}"
        path.write_bytes(base64.b64decode(b64))
        return self._rel(path)

    def _image_md(self, data: dict):
        for mime, ext in (("image/png", "png"), ("image/jpeg", "jpg"), ("image/gif", "gif")):
            if mime in data:
                b64 = join_source(data[mime]).strip()
                src = self._save_binary_image(b64, ext) if self.image_dir is not None \
                    else f"data:{mime};base64,{b64}"
                return f"![{ext}]({src})"
        if "image/svg+xml" in data:
            svg = join_source(data["image/svg+xml"])
            if self.image_dir is not None:
                self._img_idx += 1
                path = self.image_dir / f"{self.image_prefix}_{self._img_idx}.svg"
                path.write_text(svg, encoding="utf-8")
                return f"![svg]({self._rel(path)})"
        return None

    def render_output(self, out: dict) -> str:
        otype = out.get("output_type")
        if otype == "stream":
            text = strip_ansi(join_source(out.get("text"))).rstrip()
            return f"```text\n{text}\n```" if text else ""
        if otype in ("execute_result", "display_data"):
            data = out.get("data", {}) or {}
            img = self._image_md(data)
            if img:
                return img
            if "text/plain" in data:
                text = strip_ansi(join_source(data["text/plain"])).rstrip()
                return f"```text\n{text}\n```" if text else ""
            if self.include_html and "text/html" in data:
                return join_source(data["text/html"]).strip()
            return ""
        if otype == "error":
            tb = strip_ansi("\n".join(out.get("traceback", []))).rstrip()
            return f"```text\n{tb}\n```" if tb else ""
        return ""

    def convert(self, nb: dict) -> str:
        lang = ((nb.get("metadata", {}) or {}).get("language_info", {}) or {}) \
            .get("name", "python")
        lines: list[str] = []

        def add(block: str):
            if not block:
                return
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(block)
            lines.append("")

        for cell in nb.get("cells", []):
            ctype = cell.get("cell_type")
            src = join_source(cell.get("source")).rstrip()
            if ctype in ("markdown", "raw"):
                add(src)
            elif ctype == "code":
                if src:
                    add(f"```{lang}\n{src}\n```")
                if self.include_output:
                    for out in cell.get("outputs", []) or []:
                        add(self.render_output(out))
        return "\n".join(lines).strip() + "\n"


def convert_one(nb_path: Path, out_path: Path, **kwargs) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conv = NotebookConverter(out_path, **kwargs)
    out_path.write_text(conv.convert(nb), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 Jupyter Notebook 转成 Markdown（支持批量）")
    ap.add_argument("notebooks", nargs="*", help="输入的 .ipynb 文件（可多个）；不填则自动转换 --dir 下的所有 ipynb")
    ap.add_argument("--dir", default=".", help="要扫描的文件夹，默认当前目录")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子文件夹")
    ap.add_argument("-o", "--output", help="输出 .md 路径（仅单个输入时可用）")
    ap.add_argument("--no-output", action="store_true", help="不包含代码执行结果")
    ap.add_argument("--image-dir", help="图片保存目录（默认 base64 内联）")
    ap.add_argument("--image-prefix", default="image", help="图片文件名前缀")
    ap.add_argument("--include-html", action="store_true", help="保留 HTML 输出")
    ap.add_argument("--suffix", default=".md", help="输出文件后缀，默认 .md")
    args = ap.parse_args(argv)

    # ---------- 决定要处理哪些文件 ----------
    if args.notebooks:
        nb_files = [Path(p) for p in args.notebooks]
    else:
        base = Path(args.dir)
        if not base.exists():
            print(f"[错误] 文件夹不存在: {base}", file=sys.stderr)
            sys.exit(1)
        pattern = "**/*.ipynb" if args.recursive else "*.ipynb"
        nb_files = sorted(base.glob(pattern))
        # 排除 checkpoint 里的临时文件
        nb_files = [p for p in nb_files if ".ipynb_checkpoints" not in p.parts]

    if not nb_files:
        print("没有找到任何 .ipynb 文件。", file=sys.stderr)
        return

    if args.output and len(nb_files) > 1:
        ap.error("-o/--output 只能用于单个输入文件")

    print(f"共找到 {len(nb_files)} 个 notebook，开始转换...\n")

    ok, fail = 0, 0
    for nb_path in nb_files:
        if not nb_path.exists():
            print(f"[跳过] 找不到文件: {nb_path}")
            fail += 1
            continue

        out_path = Path(args.output) if args.output else nb_path.with_suffix(args.suffix)

        # 每转换一个文件，图片计数器独立（避免文件名冲突）
        image_dir = None
        if args.image_dir:
            image_dir = Path(args.image_dir)
            if not image_dir.is_absolute():
                image_dir = out_path.parent / image_dir
            image_dir.mkdir(parents=True, exist_ok=True)

        try:
            convert_one(
                nb_path, out_path,
                include_output=not args.no_output,
                image_dir=image_dir,
                image_prefix=args.image_prefix,
                include_html=args.include_html,
            )
            print(f"  ✓ {nb_path.name}  ->  {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {nb_path.name}  转换失败: {e}")
            fail += 1

    print(f"\n完成：成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    main()