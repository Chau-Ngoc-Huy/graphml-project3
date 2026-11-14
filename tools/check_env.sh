#!/usr/bin/env bash
# Kiểm tra môi trường trước khi chạy: python/venv, pytest, git, gh, kaggle (đăng nhập), GPU, dung lượng đĩa.
ok() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad() { printf "  \033[31m✗\033[0m %s\n" "$1"; }
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
{ [ -x "$PY" ] || command -v "$PY" >/dev/null; } && ok "python: $($PY --version)" || bad "thiếu .venv: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
$PY -c "import pytest, yaml" 2>/dev/null && ok "pytest, pyyaml" || bad "pip install -e '.[dev]'"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && ok "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'chưa commit')" || bad "chưa git init"
command -v gh >/dev/null && gh auth status >/dev/null 2>&1 && ok "gh đã đăng nhập" || bad "gh chưa đăng nhập (gh auth login) — chỉ cần khi tạo repo GitHub"
K=${KAGGLE:-.venv/bin/kaggle}; [ -x "$K" ] || K=kaggle
if command -v "$K" >/dev/null 2>&1 || [ -x "$K" ]; then
  "$K" datasets list --mine >/dev/null 2>&1 && ok "kaggle CLI đã xác thực" || bad "kaggle chưa xác thực: $K auth login (hoặc ~/.kaggle/kaggle.json)"
else bad "thiếu kaggle CLI (pip install kaggle)"; fi
[ -f local.mk ] && ok "local.mk: $(grep -E '^(LAB_USER|KAGGLE_USER)' local.mk | tr '\n' ' ')" || bad "thiếu local.mk (cp local.mk.example local.mk)"
command -v nvidia-smi >/dev/null && ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" || echo "  · không có GPU local (chạy trên Kaggle)"
V=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/vendor.yaml'))['name'])" 2>/dev/null)
[ -n "$V" ] && [ "$V" != "VENDOR_NAME" ] && { [ -n "$(ls -A vendor/$V 2>/dev/null)" ] && ok "vendor/$V có code" || bad "vendor/$V trống: make vendor"; }
df -h . | awk 'NR==2{print "  · đĩa trống: "$4}'
