import argparse
import os
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM
import time

# The ckpt stores hyperparameters as argparse.Namespace; torch >= 2.6 defaults
# torch.load to weights_only=True and rejects it unless allowlisted.
torch.serialization.add_safe_globals([argparse.Namespace])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    if os.environ.get('SEED'):
        torch.manual_seed(int(os.environ['SEED']))
        torch.npu.manual_seed(int(os.environ['SEED']))

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    time_total = 0.0
    test_range = int(os.environ.get('BENCH_ROUNDS','22'))
    skip = 2
    for run_idx in range(test_range):
        molecules = []
        start_time = time.time()
        for i in range(args.n_samples // args.batch_size):
            molecules_batch = model.generate_ligands(
                args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
                num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                resamplings=args.resamplings, jump_length=args.jump_length,
                timesteps=args.timesteps)
            molecules.extend(molecules_batch)

        # Make SDF files
        end_time = time.time()
        if run_idx >= skip:
            time_total += end_time - start_time
        print(f"[Timing] this round use time: {end_time - start_time:.2f} 秒")

    print(f"[Timing] 平均耗时: {time_total/(test_range - skip):.2f} 秒")
    utils.write_sdf_file(args.outfile, molecules)
