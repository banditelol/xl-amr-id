import pandas as pd
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from tqdm.auto import tqdm
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('parse_smatch_finegrained.py')
    parser.add_argument('finegrained_file', help='finegrained file to parse.')
    parser.add_argument('amr_file', help='amr file to parse.')
    parser.add_argument('--out_file', help='file to parse.', default="")
    parser.add_argument('--raw_count', help='file to parse.', default=False, action="store_true")
    args = parser.parse_args()
    out_file = args.out_file
    if args.out_file == "": 
        out_file=args.finegrained_file+".csv"
        print(f"--out_file is not passed, setting output to {out_file} instead")
    summary_list = []

    with open(args.finegrained_file, "r") as f:
        for amr in tqdm(AMRIO.read(args.amr_file)):
            summary={}
            summary["id"] = repr(amr.id.split("_")[0].split(".")[0])
            summary["sentence"] = repr(amr.sentence)
            summary["graph"] = repr(str(amr.graph))
            if not args.raw_count:
                summary["P"] = float(f.readline().strip().split()[-1])*100
                summary["R"] = float(f.readline().strip().split()[-1])*100
                summary["F"] = float(f.readline().strip())*100
            else:
                summary["best"] = int(f.readline().strip())
                summary["test"] = int(f.readline().strip())
                summary["gold"] = int(f.readline().strip())
            summary_list.append(summary)
    pd.DataFrame(summary_list).round(2).to_csv(out_file, sep=";")