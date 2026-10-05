import argparse
import json
import sys
import time
import warnings

import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from gen_haa4b_processor  import GenHaa4bProcessor


warnings.filterwarnings("ignore", message="Missing cross-reference index")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="dataset JSON")
    parser.add_argument("--dataset", required=True, help="key in the JSON")
    parser.add_argument("--job-index", required=True, type=int)
    parser.add_argument("--output", required=True, help="output ROOT file")
    parser.add_argument(
        "--use-genweight", action="store_true",
        help="fill with NanoAOD genWeight instead of unit event weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.json) as stream:
        datasets = json.load(stream)

    if args.dataset not in datasets:
        raise KeyError(f"Dataset '{args.dataset}' is not in {args.json}")

    files = datasets[args.dataset]["files"]
    if not 0 <= args.job_index < len(files):
        raise IndexError(
            f"job-index {args.job_index} outside valid range 0..{len(files)-1}"
        )

    input_file = files[args.job_index]
    print(f"[INFO] Processing {args.job_index + 1}/{len(files)}: {input_file}")

    events = None
    for attempt in range(1, 6):
        try:
            events = NanoEventsFactory.from_root(
                input_file,
                treepath="Events",
                schemaclass=NanoAODSchema,
                uproot_options={"timeout": 600},
            ).events()
            break
        except Exception as error:
            print(f"[WARNING] Attempt {attempt}/5 failed: {error}")
            if attempt == 5:
                print("[ERROR] Could not open the input file")
                return 1
            time.sleep(10)

    result = GenHaa4bProcessor(
        use_genweight=args.use_genweight
    ).process(events)

    # uproot writes hist.Hist objects as ROOT TH1/TH2 objects, including Sumw2.
    with uproot.recreate(args.output) as root_file:
        for name, histogram in result.items():
            root_file[name] = histogram

    print(f"[INFO] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

