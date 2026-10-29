import argparse
import os
import sys

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#run as a plain script (a package named kaggle would clash with the kaggle api module), so add the repo root
sys.path.insert(0, REPO)

from common.sync import PURGE_MIN_AGE, make_api, orphan_lfs_files


def main(args):
    api= make_api()

    files= list(api.list_lfs_files(args.sync_repo))
    orphans= orphan_lfs_files(api, args.sync_repo, min_age=args.min_age_minutes * 60)

    total= sum(f.size for f in files)
    freed= sum(f.size for f in orphans)

    print(f"lfs storage: {total / 1e9:.2f} GB in {len(files)} files")
    print(f"still on a branch: {(total - freed) / 1e9:.2f} GB")
    print(f"unreferenced (pruned checkpoints): {freed / 1e9:.2f} GB in {len(orphans)} files")

    for f in sorted(orphans, key=lambda f: f.pushed_at):
        print(f"  {f.pushed_at:%Y-%m-%d %H:%M}  {f.size / 1e9:6.2f} GB  {f.filename}")

    if not args.delete:
        print("dry run: nothing deleted. rerun with --delete to permanently delete the unreferenced files")
        return 0

    if orphans:
        api.permanently_delete_lfs_files(args.sync_repo, orphans, rewrite_history=False)
        print(f"permanently deleted {len(orphans)} files ({freed / 1e9:.2f} GB)")

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--sync_repo",
        type=str,
        default="zeeba1705/atml-pa3"
    )
    parser.add_argument(
        "--min_age_minutes",
        type=float,
        default=PURGE_MIN_AGE / 60
    )
    parser.add_argument(
        "--delete",
        action="store_true"
    )

    sys.exit(main(parser.parse_args()))
