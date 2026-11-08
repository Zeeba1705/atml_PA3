import argparse
import json
import os
import shutil
from types import SimpleNamespace

import pytest

import common.sync as sync_mod
import task1.train as train
from common.checkpoint import save_checkpoint
from common.sync import pull_latest_checkpoint, start_sync, sync_put, sync_wait


class FakeHub:
    #stands in for HfApi: files live in a local folder per branch, failures can be injected
    def __init__(self, root, fail_times=0):
        self.root= root
        self.fail_times= fail_times
        self.calls= []

    def path(self, branch, path_in_repo=""):
        return os.path.join(self.root, branch, path_in_repo)

    def maybe_fail(self):
        if self.fail_times > 0:
            self.fail_times-= 1
            raise ConnectionError("fake network error")

    def create_repo(self, repo_id, private, exist_ok):
        os.makedirs(self.root, exist_ok=True)

    def create_branch(self, repo_id, branch, exist_ok):
        os.makedirs(self.path(branch), exist_ok=True)

    def delete_branch(self, repo_id, branch):
        shutil.rmtree(self.path(branch), ignore_errors=True)

    def upload_folder(self, repo_id, folder_path, path_in_repo, revision, commit_message):
        self.maybe_fail()
        self.calls.append(("upload_folder", path_in_repo))
        shutil.copytree(folder_path, self.path(revision, path_in_repo), dirs_exist_ok=True)

    def upload_file(self, repo_id, path_or_fileobj, path_in_repo, revision, commit_message):
        self.maybe_fail()
        self.calls.append(("upload_file", path_in_repo))
        os.makedirs(os.path.dirname(self.path(revision, path_in_repo)), exist_ok=True)
        shutil.copy(path_or_fileobj, self.path(revision, path_in_repo))

    def delete_folder(self, repo_id, path_in_repo, revision, commit_message):
        self.calls.append(("delete_folder", path_in_repo))
        shutil.rmtree(self.path(revision, path_in_repo))

    def super_squash_history(self, repo_id, branch):
        self.calls.append(("squash", branch))

    def list_repo_files(self, repo_id, revision):
        base= self.path(revision)
        files= []
        for root, _, names in os.walk(base):
            for name in names:
                files.append(os.path.relpath(os.path.join(root, name), base))
        return files

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(branches=[SimpleNamespace(name=b) for b in os.listdir(self.root)])

    def list_repo_tree(self, repo_id, revision, recursive):
        return [SimpleNamespace(path=f, lfs=None) for f in self.list_repo_files(repo_id, revision)]

    def list_lfs_files(self, repo_id):
        #the fake has no lfs store: nothing to purge
        return []

    def permanently_delete_lfs_files(self, repo_id, lfs_files, rewrite_history):
        self.calls.append(("purge_lfs", len(list(lfs_files))))

    def hf_hub_download(self, repo_id, filename, revision, local_dir):
        src= self.path(revision, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(filename)
        dst= os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        return dst


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(sync_mod, "RETRY_SLEEP", 0)


def test_upload_retries_then_succeeds(tmp_path):
    hub= FakeHub(str(tmp_path / "hub"), fail_times=2)
    sync= start_sync("me/repo", "task1-x", api=hub)

    path= tmp_path / "f.txt"
    path.write_text("hello")

    sync_put(sync, "upload_file", str(path), "a/f.txt")

    assert sync_wait(sync)
    assert sync["failures"] == []
    assert (tmp_path / "hub" / "task1-x" / "a" / "f.txt").read_text() == "hello"


def test_upload_failure_is_logged_not_raised(tmp_path):
    hub= FakeHub(str(tmp_path / "hub"), fail_times=100)
    sync= start_sync("me/repo", "task1-x", api=hub)

    path= tmp_path / "f.txt"
    path.write_text("hello")

    sync_put(sync, "upload_file", str(path), "a/f.txt")
    sync_put(sync, "upload_file", str(path), "a/g.txt")

    #training would keep going: wait returns False, failures are recorded, nothing raised
    assert not sync_wait(sync)
    assert len(sync["failures"]) == 2


def test_sync_disabled_when_hub_unreachable(tmp_path):
    class DeadHub(FakeHub):
        def create_repo(self, repo_id, private, exist_ok):
            raise ConnectionError("no network")

    assert start_sync("me/repo", "task1-x", api=DeadHub(str(tmp_path))) is None
    assert sync_wait(None)


def fake_remote_ckpt(hub, branch, prefix, kind, update, tmp):
    local= os.path.join(tmp, "make")
    path= save_checkpoint(local, kind, update, {"trainable.pt": {"u": update}}, {"config_hash": "x"})
    shutil.copytree(path, hub.path(branch, f"{prefix}/{kind}/ckpt_{update:05d}"))


def test_pull_takes_newest_valid_remote_checkpoint(tmp_path):
    hub= FakeHub(str(tmp_path / "hub"))
    prefix= "outputs/task1/run"

    fake_remote_ckpt(hub, "b", prefix, "rolling", 50, str(tmp_path))
    fake_remote_ckpt(hub, "b", prefix, "rolling", 100, str(tmp_path))

    #newest remote one is corrupt -> falls back to 50
    with open(hub.path("b", f"{prefix}/rolling/ckpt_00100/trainable.pt"), "ab") as f:
        f.write(b"junk")

    run_dir= str(tmp_path / "local" / "run")
    staging= str(tmp_path / "staging")

    pulled= pull_latest_checkpoint(hub, "me/repo", "b", prefix, run_dir, -1, staging)
    assert pulled == os.path.join(run_dir, "rolling", "ckpt_00050")
    assert os.path.exists(os.path.join(pulled, "manifest.json"))

    #local already has 50 -> nothing newer to pull
    assert pull_latest_checkpoint(hub, "me/repo", "b", prefix, run_dir, 50, staging) is None


def train_args(root, **extra):
    args= {
        "config": "configs/task1/full_ft.yaml",
        "run_name": "synced",
        "rank": None,
        "alpha": None,
        "smoke": True,
        "fresh": False,
        "force": False,
        "device": "cpu",
        "output_dir": os.path.join(root, "outputs"),
        "results_dir": os.path.join(root, "results"),
        "sync_repo": "me/atml-pa3",
        "max_hours": 11.3,
        "ckpt_every": None,
        "num_workers": 0,
        "skip_preflight": False,
        "crash_at_update": None,
        "sigterm_at_update": None,
        "nan_at_update": None
    }
    args.update(extra)

    return argparse.Namespace(**args)


def run_main(args):
    train.STOP["reason"]= None

    with pytest.raises(SystemExit) as e:
        train.main(args)

    return e.value.code


def read_rows(root):
    with open(os.path.join(root, "results", "task1", "synced", "train_log.jsonl")) as f:
        return [json.loads(line) for line in f]


def test_new_session_pulls_synced_state_and_matches(tmp_path, monkeypatch):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    straight= str(tmp_path / "straight")
    session= str(tmp_path / "session")

    hub_a= FakeHub(str(tmp_path / "hub_a"))
    monkeypatch.setattr(sync_mod, "make_api", lambda: hub_a)
    assert run_main(train_args(straight)) == 0

    #session 1 stops after one update (max_hours=0) and syncs
    hub_b= FakeHub(str(tmp_path / "hub_b"))
    monkeypatch.setattr(sync_mod, "make_api", lambda: hub_b)
    assert run_main(train_args(session, max_hours=0)) == 0

    #kaggle wipes /kaggle/working: the next session starts with nothing local
    shutil.rmtree(os.path.join(session, "outputs"))
    shutil.rmtree(os.path.join(session, "results"))

    assert run_main(train_args(session)) == 0

    assert read_rows(session) == read_rows(straight)

    #a finished full-ft run keeps only its exports, locally and on the remote (states deleted, branch squashed)
    remote= hub_b.list_repo_files("me/atml-pa3", "task1-synced")
    assert [f for f in remote if f.endswith("manifest.json")] == []
    assert "outputs/task1/synced/export/update_00020/model.safetensors" in remote
    assert "results/task1/synced/train_summary.json" in remote
    assert ("squash", "task1-synced") in hub_b.calls

    #locally only the exports are left (the checkpoint folders are empty)
    run_dir= os.path.join(session, "outputs", "task1", "synced")
    for kind in ["rolling", "milestones"]:
        assert os.listdir(os.path.join(run_dir, kind)) == []
    assert sorted(os.listdir(os.path.join(run_dir, "export"))) == ["update_00010", "update_00020"]


def test_orphan_lfs_files_keeps_referenced_and_recent(tmp_path):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace as NS

    now= datetime.now(timezone.utc)
    old= now - timedelta(hours=2)

    class LfsHub:
        def list_repo_refs(self, repo_id):
            return NS(branches=[NS(name="main"), NS(name="task1-a")])

        def list_repo_tree(self, repo_id, revision, recursive):
            tips= {"main": [NS(path="README.md", lfs=None)], "task1-a": [NS(path="ckpt/state.pt", lfs=NS(sha256="live"))]}
            return tips[revision]

        def list_lfs_files(self, repo_id):
            return [
                NS(file_oid="live", size=7, pushed_at=old),
                NS(file_oid="pruned", size=7, pushed_at=old),
                NS(file_oid="uploading", size=7, pushed_at=now)
            ]

    orphans= sync_mod.orphan_lfs_files(LfsHub(), "me/repo")

    #only the old unreferenced object; the one on a branch and the fresh upload are kept
    assert [f.file_oid for f in orphans] == ["pruned"]


def test_no_remote_delete_after_failed_upload(tmp_path):
    #the upload of the new checkpoint fails: the old one must stay on the remote
    hub= FakeHub(str(tmp_path / "hub"))
    sync= start_sync("me/repo", "task1-x", api=hub)

    path= tmp_path / "f.txt"
    path.write_text("hello")
    sync_put(sync, "upload_file", str(path), "old/f.txt")
    assert sync_wait(sync)

    hub.fail_times= 100
    sync_put(sync, "upload_file", str(path), "new/f.txt")
    sync_put(sync, "delete_folder", "old")
    sync_wait(sync)

    assert (tmp_path / "hub" / "task1-x" / "old" / "f.txt").exists()
    assert ("delete_folder", "old") not in hub.calls
