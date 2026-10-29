import os
import queue
import re
import shutil
import threading
import time

from huggingface_hub import HfApi

from common.checkpoint import verify_checkpoint

RETRIES= 3
RETRY_SLEEP= 10
WAIT_LIMIT= 45 * 60
CKPT_FILE= re.compile(r"^(.*)/(rolling|milestones)/ckpt_(\d+)/([^/]+)$")


def make_api():
    #token comes from kaggle secrets as HF_TOKEN
    return HfApi(token=os.environ.get("HF_TOKEN"))


def start_sync(repo_id, branch, api=None):
    #one branch per job: two jobs never write the same branch, and squashing one can't touch the other
    api= api or make_api()

    try:
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.create_branch(repo_id, branch=branch, exist_ok=True)
    except Exception as e:
        print(f"WARNING sync disabled, cannot reach {repo_id}: {e}")
        return None

    sync= {
        "api": api,
        "repo_id": repo_id,
        "branch": branch,
        "queue": queue.Queue(),
        "failures": [],
        "uploaded": 0
    }

    thread= threading.Thread(target=sync_worker, args=(sync,), daemon=True)
    thread.start()

    return sync


def run_op(sync, op):
    api= sync["api"]
    kind= op[0]

    if kind == "upload_folder":
        api.upload_folder(
            repo_id=sync["repo_id"],
            folder_path=op[1],
            path_in_repo=op[2],
            revision=sync["branch"],
            commit_message=f"upload {op[2]}"
        )
    elif kind == "upload_file":
        api.upload_file(
            repo_id=sync["repo_id"],
            path_or_fileobj=op[1],
            path_in_repo=op[2],
            revision=sync["branch"],
            commit_message=f"upload {op[2]}"
        )
    elif kind == "delete_folder":
        api.delete_folder(
            repo_id=sync["repo_id"],
            path_in_repo=op[1],
            revision=sync["branch"],
            commit_message=f"delete {op[1]}"
        )
    elif kind == "squash":
        #deleted checkpoints stay in git history and count as storage until the branch is squashed
        api.super_squash_history(repo_id=sync["repo_id"], branch=sync["branch"])
    else:
        raise ValueError(f"unknown sync op {kind}")


def sync_worker(sync):
    #ops run in order on one thread; an op that keeps failing is logged and skipped, never raised
    while True:
        op= sync["queue"].get()

        for attempt in range(RETRIES):
            try:
                run_op(sync, op)
                sync["uploaded"]+= 1
                print(f"sync ok: {op[0]} {op[-1]}")
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    print(f"WARNING sync failed after {RETRIES} tries: {op[0]} {op[-1]}: {e}")
                    sync["failures"].append((op[0], op[-1], str(e)))
                else:
                    time.sleep(RETRY_SLEEP * (attempt + 1))

        sync["queue"].task_done()


def sync_put(sync, *op):
    if sync is not None:
        sync["queue"].put(op)


def sync_wait(sync, limit=WAIT_LIMIT):
    #block until queued uploads are done (or give up after limit seconds, with a warning)
    if sync is None:
        return True

    start= time.time()
    while sync["queue"].unfinished_tasks > 0:
        if time.time() - start > limit:
            print("WARNING sync still uploading after", limit, "s, giving up waiting")
            return False
        time.sleep(1)

    return len(sync["failures"]) == 0


def remote_checkpoints(api, repo_id, branch, run_prefix):
    #{update: (kind, [files])} for checkpoints under run_prefix on the branch
    try:
        files= api.list_repo_files(repo_id, revision=branch)
    except Exception as e:
        print(f"WARNING cannot list {repo_id}@{branch}: {e}")
        return {}

    found= {}
    for path in files:
        match= CKPT_FILE.match(path)
        if match and match.group(1) == run_prefix:
            update= int(match.group(3))
            found.setdefault(update, (match.group(2), []))[1].append(path)

    return found


def pull_latest_checkpoint(api, repo_id, branch, run_prefix, run_dir, local_update, staging):
    #download the newest remote checkpoint that is newer than local_update and passes its manifest check
    remote= remote_checkpoints(api, repo_id, branch, run_prefix)

    for update in sorted(remote, reverse=True):
        if update <= local_update:
            return None

        kind, files= remote[update]

        if os.path.exists(staging):
            shutil.rmtree(staging)

        try:
            for path in files:
                api.hf_hub_download(repo_id, path, revision=branch, local_dir=staging)
        except Exception as e:
            print(f"WARNING download of remote ckpt_{update:05d} failed: {e}")
            continue

        src= os.path.join(staging, run_prefix, kind, f"ckpt_{update:05d}")
        ok, reason= verify_checkpoint(src)
        if not ok:
            print(f"remote ckpt_{update:05d} is corrupt ({reason}), trying the previous one")
            continue

        dst= os.path.join(run_dir, kind, f"ckpt_{update:05d}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        shutil.rmtree(staging)

        print(f"pulled remote checkpoint at update {update}")
        return dst

    return None


def pull_file(api, repo_id, branch, path_in_repo, local_path, staging):
    try:
        api.hf_hub_download(repo_id, path_in_repo, revision=branch, local_dir=staging)
    except Exception:
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.move(os.path.join(staging, path_in_repo), local_path)

    return True
