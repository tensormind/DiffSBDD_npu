#!/bin/bash
# Fixed 4-config device-level bisection of the v0-vs-v1 gap (PERSTEP device ms/step).
# Configs (all SEED=3 for E-matched, BENCH_ROUNDS=4 -> 2 counted rounds):
#   A: clones back on      V0_CLONES=1 V0_V1MASK=0 V0_NAGATHER=0
#   B: default (current)   (all unset)                <- optimized v0 baseline
#   C: v1-style 1-pair mask V0_V1MASK=1
#   D: index_select gathers V0_NAGATHER=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/workspace/v0/DiffSBDD

run_cfg() {
    local name=$1; shift
    local log=/tmp/bis_conf_${name}.log
    echo "=== [$name] $(date +%H:%M:%S) start ===" | tee "$log"
    env -u PYTHONPATH SEED=3 PERSTEP=1 BENCH_ROUNDS=4 "$@" \
        /root/miniconda3/envs/diffsbdd/bin/python ab_bench.py \
        checkpoints/crossdocked_fullatom_cond.ckpt \
        --pdbfile pdb_files/164707_1749718027_dMJmu.pdb \
        --resi_list "A:300" --outfile /tmp/out_${name}.sdf \
        >> "$log" 2>&1
    echo "=== [$name] $(date +%H:%M:%S) end (rc=$?) ===" >> "$log"
}

run_cfg A V0_CLONES=1 V0_V1MASK=0 V0_NAGATHER=0
run_cfg B
run_cfg C V0_V1MASK=1 V0_NAGATHER=0
run_cfg D V0_NAGATHER=1 V0_CLONES=0 V0_V1MASK=0

echo "=== ALL DONE $(date +%H:%M:%S) ==="