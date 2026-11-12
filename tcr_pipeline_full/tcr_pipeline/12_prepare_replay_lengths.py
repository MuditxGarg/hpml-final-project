"""Reconstruct the corrected marginal length pools used by the matched E1 replay."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare_lengths(run, cap=2048):
    run = Path(run)
    original_files = sorted((run/'rollouts').glob('chunk_*.json'))
    cap_files = sorted((run/'rollouts_nothink_cap').glob('chunk_*.json'))
    original = [row for f in original_files for row in json.loads(f.read_text())]
    corrected_rows = [row for f in cap_files for row in json.loads(f.read_text())]
    corrected = {row['id']:row for row in corrected_rows}
    ids = [row['id'] for row in original]
    if not original or len(set(ids)) != len(ids) or len(corrected) != len(corrected_rows) or set(ids) != set(corrected):
        raise ValueError('Original and corrected records must uniquely cover the same questions')
    lengths = {'0':[], '1024':[], '2048':[]}
    for row in original:
        samples = corrected[row['id']]['samples']
        if len(samples) != len(row['traces']):
            raise ValueError('Corrected and original sample counts differ')
        for trace, sample in zip(row['traces'],samples):
            lengths['0'].append(min(sample['ft'],cap))
            for budget in (1024,2048):
                lengths[str(budget)].append(min(trace['L'],budget)+trace['b'][str(budget)]['ft'])
    if cap <= 0 or any(x <= 0 for pool in lengths.values() for x in pool):
        raise ValueError('Replay lengths and cap must be positive')
    return lengths, dict(cap=cap,questions=len(original),samples_per_budget={b:len(pool) for b,pool in lengths.items()},
        ordering='lexically sorted chunk files; original question and trace order within each file',
        definition='B0=min(corrected final tokens,cap); other B=min(original thinking length,B)+final tokens',
        limitation='marginal completion-length pools; prompt-length/correctness correlations are not preserved by replay',
        files_sha256={str(f.relative_to(run)):hashlib.sha256(f.read_bytes()).hexdigest() for f in original_files+cap_files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).parent/'tcr_results/main')
    parser.add_argument('--cap',type=int,default=2048)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    lengths,manifest=prepare_lengths(args.run,args.cap)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'lengths.json').write_text(json.dumps(lengths))
    manifest['lengths_file_sha256']=hashlib.sha256((args.output/'lengths.json').read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
