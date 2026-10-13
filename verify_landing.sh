#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY=/root/miniconda3/envs/diffsbdd/bin/python
cd /root/workspace/v0/DiffSBDD

echo "########## A: V0_NAGATHER=0 (advanced indexing, regression) ##########"
env -u PYTHONPATH SEED=3 PERSTEP=1 BENCH_ROUNDS=4 V0_NAGATHER=0 \
    $PY ab_bench.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile pdb_files/164707_1749718027_dMJmu.pdb --resi_list "A:300" \
    --outfile /tmp/v0_gatherOFF.sdf 2>&1 \
    | grep -E '\[Timing\]|PERSTEP_device|Traceback'; echo "rc_A=${PIPESTATUS[0]}"

echo "########## B: default env (index_select ON, 落地状态) ##########"
env -u PYTHONPATH SEED=3 PERSTEP=1 BENCH_ROUNDS=4 \
    $PY ab_bench.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile pdb_files/164707_1749718027_dMJmu.pdb --resi_list "A:300" \
    --outfile /tmp/v0_gatherON.sdf 2>&1 \
    | grep -E '\[Timing\]|PERSTEP_device|Traceback'; echo "rc_B=${PIPESTATUS[0]}"

echo "########## C: 官方命令（默认环境，无任何 env） ##########"
env -u PYTHONPATH \
    $PY generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile pdb_files/164707_1749718027_dMJmu.pdb --resi_list "A:300" \
    --outfile /tmp/v0_official_default.sdf --n_samples 20 --batch_size 20 2>&1 \
    | grep -E '\[Timing\]|Traceback|训练|输出|耗时'; echo "rc_C=${PIPESTATUS[0]}"
echo "=== DONE $(date +%H:%M:%S) ==="