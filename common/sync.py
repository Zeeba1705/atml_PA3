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
#an unreferenced lfs object younger than this may be an upload whose commit hasn't landed yet: never purge it
PURGE_MIN_AGE= 30 * 60


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
    elif kind in ["delete_folder", "delete_finished"]:
        api.delete_folder(
            repo_id=sync["repo_id"],
            path_in_repo=op[1],
            revision=sync["branch"],
            commit_message=f"delete {op[1]}"
        )
    elif kind == "squash":
        #deleted checkpoints stay in git history and count as storage until the branch is squashed
        api.super_squash_history(repo_id=sync["repo_id"], branch=sync["branch"])
    elif kind == "purge":
        #even after squashing, the hub keeps (and bills) the lfs objects until they are permanently deleted
        orphans= orphan_lfs_files(api, sync["repo_id"])
        if orphans:
            api.permanently_delete_lfs_files(sync["repo_id"], orphans, rewrite_history=False)
            print(f"purged {len(orphans)} unreferenced lfs files ({sum(f.size for f in orphans) / 1e9:.2f} GB)")
    else:
        raise ValueError(f"unknown sync op {kind}")


def referenced_oids(api, repo_id):
    #sha256 of every lfs file at the tip of any branch
    oids= set()

    for ref in api.list_repo_refs(repo_id).branches:
        for entry in api.list_repo_tree(repo_id, revision=ref.name, recursive=True):
            lfs= getattr(entry, "lfs", None)
            if lfs is not None:
                oids.add(lfs.sha256)

    return oids


def orphan_lfs_files(api, repo_id, min_age=PURGE_MIN_AGE):
    #lfs objects no branch points to any more (pruned checkpoints), old enough not to be a pending upload.
    #references are listed first, so a file committed in between is caught by the age check instead
    referenced= referenced_oids(api, repo_id)
    now= time.time()

    return [
        f for f in api.list_lfs_files(repo_id)
        if f.file_oid not in referenced and now - f.pushed_at.timestamp() > min_age
    ]


def sync_worker(sync):
    #ops run in order on one thread; an op that keeps failing is logged and skipped, never raised
    while True:
        op= sync["queue"].get()

        #after a failed upload the remote may lack the newest checkpoint: keep the older ones it still has.
        #delete_finished (states of a run that is done) doesn't need them
        if op[0] == "delete_folder" and any(f[0].startswith("upload") for f in sync["failures"]):
            print(f"WARNING skipping remote delete of {op[1]}: an earlier upload failed")
            sync["queue"].task_done()
            continue

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
