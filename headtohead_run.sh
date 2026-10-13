#!/bin/bash
# Same-session gold-standard: v1 (6 rounds) then v0+configD (6 rounds), SEED=3.
# Compare per-round wall + E per ring; E matches at least the first 3 rings.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "########## V1 (6 rounds) ##########"
cd /root/workspace/v1/DiffSBDD
env -u PYTHONPATH SEED=3 AB_TIMER=1 BENCH_ROUNDS=6 \
    /root/miniconda3/envs/diffsbdd/bin/python generate_ligands.py \
    checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile pdb_files/164707_1749718027_dMJmu.pdb \
    --resi_list "A:300" --outfile /tmp/v1_headtohead.sdf 2>&1 \
    | grep -E '\[AB\] loop|ExactBudget|平均|Timing' ; echo "rc_v1=${PIPESTATUS[0]}"

echo "########## V0 + config D (6 rounds) ##########"
cd /root/workspace/v0/DiffSBDD
env -u PYTHONPATH SEED=3 PERSTEP=1 BENCH_ROUNDS=6 V0_NAGATHER=1 \
    /root/miniconda3/envs/diffsbdd/bin/python ab_bench.py \
    checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile pdb_files/164707_1749718027_dMJmu.pdb \
    --resi_list "A:300" --outfile /tmp/v0D_headtohead.sdf 2>&1 \
    | grep -E '\[Timing\]|tables \{|PERSTEP_device' ; echo "rc_v0=${PIPESTATUS[0]}"
echo "=== HEAD-TO-HEAD DONE $(date +%H:%M:%S) ==="