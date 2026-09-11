set -u
PY="C:/Users/36852/AppData/Local/Programs/Python/Python310/python.exe"
LOG="out/_batch.log"
: > "$LOG"
run () {
  echo "=== $* ===" >> "$LOG"
  "$PY" "$@" >> "$LOG" 2>&1
  echo "--- exit=$? ---" >> "$LOG"
}
run run_stage.py  --problem Q3    --ladder legacy-greedy,fixed-greedy,fixed-mpc,exact-greedy
run run_stage.py  --problem Q4-3  --ladder legacy-greedy,fixed-greedy,fixed-mpc,exact-greedy
run run_single.py --problem Q2    --ladder legacy-greedy,reserve-greedy,reserve-mpc --perfect
run run_single.py --problem Q4-2  --ladder legacy-greedy,reserve-greedy,reserve-mpc --perfect
run run_single.py --problem Q4-2  --ladder legacy-greedy,reserve-greedy,reserve-mpc --price-perfect
run run_stage.py  --problem Q3    --ladder legacy-greedy,fixed-greedy,exact-greedy --perfect
run run_stage.py  --problem Q4-3  --ladder legacy-greedy,fixed-greedy,exact-greedy --perfect
echo "ALL-DONE" >> "$LOG"
