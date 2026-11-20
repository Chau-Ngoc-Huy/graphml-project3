"""Đồng bộ thay đổi của nhóm từ `main` (cấu trúc cũ: huy/ thang/ khanh/ duong/ yen/ shared/) sang nhánh `final` (cấu trúc mới).

    python tools/sync_main.py --dry-run          # chỉ liệt kê: commit mới, file nào tự sang chỗ mới, file nào phải chuyển tay
    python tools/sync_main.py                    # fetch + merge origin/main + kiểm tra sau merge + chạy test
    python tools/sync_main.py --ref sim-main     # đồng bộ từ một ref khác (thử nghiệm)

Cách hoạt động (docs/migration-map.tsv, khớp tiền tố dài nhất):
  moved      file chỉ đổi chỗ → `git merge` (nhận diện đổi tên + directoryRenames=true) tự đưa thay đổi vào đường dẫn mới,
             kể cả file MỚI tạo trong thư mục cũ đã đổi tên (vd khanh/src/x.py → src/adamf/x.py).
  rewritten  file đã gộp/viết lại trên final → merge không tự chuyển được; script in diff của main và đích cần chuyển tay.
  dropped    đã bỏ trên final → chỉ báo (kèm diff), không mang sang.
  unmapped   file mới trên main không thuộc tiền tố nào trong bảng → script báo để xếp chỗ + thêm dòng vào bảng.
Xung đột "sửa/xóa" trên file rewritten/dropped (main sửa, final đã gộp/bỏ) được giữ phía final (diff đã nằm trong báo cáo).
Sau merge: báo file nào lại xuất hiện ở thư mục cũ (cần git mv), chạy pytest. Báo cáo: logs/sync/<ngày>_<sha>.md.
Không bao giờ push; không sửa lịch sử (merge thường, giữ commit gốc của từng người).
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ("huy/", "thang/", "khanh/", "duong/", "yen/", "shared/")


def git(*a, check=True):
    p = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"git {' '.join(a)} lỗi:\n{p.stderr}")
    return p.stdout


def load_map():
    rows = []
    for line in (ROOT / "docs" / "migration-map.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[1], c[2] if len(c) > 2 else "moved"))
    return sorted(rows, key=lambda r: -len(r[0]))


def locate(path, rows):
    """(đường dẫn mới, kind) cho một đường dẫn cũ; (path, 'same') nếu ngoài các thư mục cũ."""
    for old, new, kind in rows:
        if path == old or (old.endswith("/") and path.startswith(old)):
            return ("-" if new == "-" else new + path[len(old):] if old.endswith("/") else new), kind
    return (path, "unmapped") if path.startswith(OLD) else (path, "same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()

    if git("status", "--porcelain", "--untracked-files=no").strip() and not a.dry_run:
        sys.exit("Cây làm việc còn thay đổi chưa commit — commit hoặc stash trước khi đồng bộ.")
    if not a.no_fetch and a.ref.startswith("origin/"):
        git("fetch", "origin")
    rows = load_map()
    base = git("merge-base", "HEAD", a.ref).strip()
    head = git("rev-parse", "--short", a.ref).strip()
    commits = git("log", "--format=%h %an: %s", f"{base}..{a.ref}").strip().splitlines()
    changes = [l.split("\t") for l in git("diff", "--name-status", "--no-renames", base, a.ref).strip().splitlines() if l]

    rep = [f"# Đồng bộ {a.ref} ({head}) → final · {dt.datetime.now():%Y-%m-%d %H:%M}", "",
           f"Gốc chung: `{base[:9]}` · {len(commits)} commit mới, {len(changes)} file đổi.", "", "## Commit", ""]
    rep += [f"- {c}" for c in commits] or ["- (không có)"]
    groups = {"moved": [], "same": [], "rewritten": [], "dropped": [], "unmapped": []}
    for st, path in changes:
        new, kind = locate(path, rows)
        groups[kind].append((st, path, new))
    label = {"moved": "Tự sang chỗ mới khi merge", "same": "Ngoài thư mục cá nhân (giữ đường dẫn)",
             "rewritten": "PHẢI CHUYỂN TAY (file đã gộp/viết lại trên final)", "unmapped": "CHƯA CÓ CHỖ trong bảng ánh xạ",
             "dropped": "Đã bỏ trên final (chỉ báo)"}
    for k in ("rewritten", "unmapped", "dropped", "moved", "same"):
        if groups[k]:
            rep += ["", f"## {label[k]} ({len(groups[k])})", ""]
            rep += [f"- `{st}` {p}" + (f" → `{n}`" if n != p else "") for st, p, n in groups[k]]
    for st, p, n in groups["rewritten"] + groups["dropped"]:
        rep += ["", f"### diff `{p}`" + (f" (chuyển sang `{n}`)" if n != "-" else " (đã bỏ trên final)"), "", "```diff",
                git("diff", base, a.ref, "--", p).strip()[:6000], "```"]

    if not a.dry_run and commits:
        msg = f"final: đồng bộ {a.ref} @{head} ({len(commits)} commit)"
        r = subprocess.run(["git", "-c", "merge.directoryRenames=true", "-c", "merge.renames=true", "-c", "diff.renameLimit=5000",
                            "merge", "--no-ff", "-m", msg, a.ref], cwd=ROOT, capture_output=True, text=True)
        rep += ["", "## Merge", "", "```", (r.stdout + r.stderr).strip()[-4000:], "```"]
        handoff = {p for k in ("rewritten", "dropped") for _, p, _ in groups[k]}
        for f in git("diff", "--name-only", "--diff-filter=U").split():
            # sửa/xóa: main sửa file mà final đã gộp/bỏ (kể cả bản legacy/ của nó) → giữ phía final, diff đã ở báo cáo
            if (f in handoff or f.startswith("legacy/")) and subprocess.run(["git", "cat-file", "-e", f"HEAD:{f}"], cwd=ROOT,
                                                                           capture_output=True).returncode:
                git("rm", "-q", "--cached", "--ignore-unmatch", f)
                (ROOT / f).unlink(missing_ok=True)
                rep += [f"- giữ phía final cho `{f}` (đã gộp/bỏ) — chuyển tay theo diff ở trên nếu cần"]
        if r.returncode and not git("diff", "--name-only", "--diff-filter=U").strip():
            # chỉ là cảnh báo "directory rename split" (file mới trong thư mục cũ bị tách ra nhiều nơi): nội dung đã gộp sạch,
            # file mới nằm lại thư mục cũ và được liệt kê bên dưới để git mv
            git("commit", "--no-edit", "-q")
            r = subprocess.CompletedProcess(r.args, 0)
            rep += ["", "Không có file xung đột nội dung — đã commit merge; xem mục file nằm ở thư mục cũ."]
        if r.returncode:
            rep += ["", "**Merge dừng vì xung đột** — sửa các file trên (`git status`), `git add`, `git commit`; "
                        "hoặc `git merge --abort` để quay lại."]
        stray = [f for f in git("ls-files").splitlines() if f.startswith(OLD)]
        if stray:
            rep += ["", "## File lại nằm ở thư mục cũ sau merge (cần `git mv` + thêm vào bảng ánh xạ)", ""]
            rep += [f"- {f} → `{locate(f, rows)[0]}`" for f in stray]
        if not r.returncode and not a.no_test:
            t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                               capture_output=True, text=True)
            rep += ["", "## Test sau merge", "", "```", t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-500:], "```"]

    out = ROOT / "logs" / "sync" / f"{dt.date.today()}_{head}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(l for l in rep if not l.startswith(("```", "-  ", "+", "@@", "diff --git", "index "))))
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
