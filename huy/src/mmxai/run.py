"""Điểm vào: python -m mmxai.run --exp <tên thí nghiệm> [--limit N] [--tag t] [--set key=value ...]

Mỗi thí nghiệm là một module src/mmxai/experiments/<tên>.py có hàm run(ctx) -> dict (xem experiments/_template.py)
và một cấu hình configs/experiments/<tên>.yaml. Harness lo: run_id, seed, thư mục runs/, metrics.json, results/.
"""

from .core.runner import main

if __name__ == "__main__":
    main()
