import os

import torch
from safetensors.torch import save_file

from task1.evaluate import INVALID, gsm8k_table, load_adapters, metric_value
from task1.merge import adapter_tensors
from task1.methods.lora import LoRALinear, add_lora
from task1.methods.oplora import add_oplora
from tests.test_task1 import TARGETS, tiny_qwen3


def fake_sample(doc_id, filt, generation, extracted, correct):
    return {
        "doc_id": doc_id,
        "doc": {"question": f"q{doc_id}"},
        "arguments": [[f"prompt {doc_id}", {}]],
        "target": "4",
        "resps": [[generation]],
        "filtered_resps": [extracted],
        "filter": filt,
        "exact_match": float(correct)
    }


def test_gsm8k_table_flags():
    samples= [
        fake_sample(1, "strict-match", "so 2 + 2 = 4", INVALID, False),
        fake_sample(1, "flexible-extract", "so 2 + 2 = 4", "4", True),
        fake_sample(0, "flexible-extract", "a b c d e", INVALID, False),
        fake_sample(2, "flexible-extract", "1 2 3", "3", False)
    ]

    #new-token counts recorded during generation, keyed by prompt
    lengths= {"prompt 0": 5, "prompt 1": 6, "prompt 2": 3}

    rows= gsm8k_table(samples, lengths, max_gen_toks=5)

    #only flexible-extract rows, sorted by id
    assert [r["id"] for r in rows] == [0, 1, 2]

    #id 0 hit the 5-token cap with no number; id 1 is right; id 2 is extracted but wrong
    assert [r["truncated"] for r in rows] == [True, True, False]
    assert [r["extract_failed"] for r in rows] == [True, False, False]
    assert [r["correct"] for r in rows] == [False, True, False]
    assert rows[1]["n_tokens"] == 6


def test_metric_value_reads_stderr():
    res= {"acc_norm,none": 0.5, "acc_norm_stderr,none": 0.01}

    assert metric_value(res, "acc_norm,none") == {"name": "acc_norm,none", "value": 0.5, "stderr": 0.01}


def test_exported_oplora_factors_rebuild_the_trained_model(tmp_path):
    #eval rebuilds every adapter run as a plain LoRALinear with the exported (projected) factors.
    #that must give exactly the logits of the trained oplora model
    trained= tiny_qwen3()
    add_oplora(trained, TARGETS, rank=4, alpha=8, dropout=0.0, seed=6304, k=4)

    with torch.no_grad():
        for m in trained.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.1)

    path= os.path.join(tmp_path, "adapters.safetensors")
    save_file(adapter_tensors(trained), path)

    rebuilt= tiny_qwen3()
    add_lora(rebuilt, TARGETS, rank=4, alpha=8, dropout=0.0, seed=6304)
    load_adapters(rebuilt, path)

    ids= torch.randint(0, 64, (2, 10), generator=torch.Generator().manual_seed(0))

    trained.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(trained(ids).logits, rebuilt(ids).logits, atol=1e-6)
