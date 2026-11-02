import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, SiglipVisionConfig, SiglipVisionModel

from common.bootstrap import paired_difference_ci, resample_indices
from common.io import load_yaml
from task2.build_data import check_isolation, grounding_pairs
from task2.connector import Connector, calibrate_scale
from task2.evaluate import generate, mismatch_map
from task2.vlm import IGNORE, build_example, collate, example_nll, prompt_segments, vlm_loss
from task2.vqa import normalize_answer, soft_accuracy

CFG= load_yaml("configs/task2/base.yaml")
#tests use a narrow llm and connector (64 wide) to stay light; the wiring does not depend on the width
TINY_DIM= 64


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CFG["llm_name"])


def tiny_llm(tok):
    #tiny random qwen3 with the real vocab and hidden size = connector output, frozen like the real backbone
    config= Qwen3Config(
        vocab_size=len(tok),
        hidden_size=TINY_DIM,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=True
    )
    torch.manual_seed(0)
    llm= Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa", dtype=torch.float32)

    for p in llm.parameters():
        p.requires_grad= False

    return llm


def make(tok, question, target, has_image, supervise_eot=True):
    return build_example(
        prompt_segments(tok, CFG),
        tok,
        question,
        target,
        has_image,
        CFG["n_patches"],
        CFG["max_question_tokens"],
        CFG["max_answer_tokens"],
        supervise_eot
    )


def test_visual_tokens_sit_in_the_user_message_before_the_question(tok):
    seg= prompt_segments(tok, CFG)
    ex= make(tok, "How many dogs can you see?", "2", True)

    start= len(seg["before"])
    vis= [i for i, v in enumerate(ex["is_visual"]) if v]

    #right after "<|im_start|>user\n", 196 of them, then "\n" + question
    assert tok.decode(ex["input_ids"][:start]) == "<|im_start|>user\n"
    assert vis == list(range(start, start + CFG["n_patches"]))
    assert tok.decode(ex["input_ids"][vis[-1] + 1:]).startswith("\nHow many dogs can you see?<|im_end|>")

    #visual tokens and the whole prompt are unlabelled; the answer and end-of-turn are labelled
    assert all(ex["labels"][i] == IGNORE for i in vis)
    labelled= [t for t in ex["labels"] if t != IGNORE]
    assert tok.decode(labelled) == "2<|im_end|>"

    #answer nll excludes the end-of-turn token
    assert tok.decode([t for t in make(tok, "q", "2", True, False)["labels"] if t != IGNORE]) == "2"


def test_no_image_input_drops_visual_tokens_and_rebuilds_mask_and_positions(tok):
    with_img= make(tok, "Is there a cat?", "yes", True)
    no_img= make(tok, "Is there a cat?", "yes", False)

    #same text, 196 fewer positions
    assert len(with_img["input_ids"]) - len(no_img["input_ids"]) == CFG["n_patches"]
    assert not any(no_img["is_visual"])

    batch= collate([with_img, no_img], tok.pad_token_id, left_pad=False)
    n= len(no_img["input_ids"])

    #the shorter row: mask covers only its tokens, positions run 0..n-1 with no gaps
    assert batch["attention_mask"][1].sum().item() == n
    assert batch["position_ids"][1, :n].tolist() == list(range(n))

    #left padding gives the same positions on the real tokens
    left= collate([with_img, no_img], tok.pad_token_id, left_pad=True)
    assert left["position_ids"][1, -n:].tolist() == list(range(n))


def test_loss_is_mean_within_example_then_across_batch(tok):
    llm= tiny_llm(tok)
    conn= Connector(CFG["vision_dim"], TINY_DIM, TINY_DIM)

    #different target lengths so per-token and per-example averaging differ
    exs= [make(tok, "Describe it.", "a dog runs on the grass near a red fence", True), make(tok, "Is it red?", "no", True)]
    batch= collate(exs, tok.pad_token_id, left_pad=False)
    feats= torch.randn(2, CFG["n_patches"], CFG["vision_dim"])

    loss= vlm_loss(llm, conn, batch, feats)

    #by hand from the full logits
    embeds= llm.get_input_embeddings()(batch["input_ids"])
    embeds[batch["is_visual"]]= conn(feats).reshape(-1, TINY_DIM)
    logits= llm(inputs_embeds=embeds, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"]).logits
    logp= F.log_softmax(logits[:, :-1].float(), dim=-1)

    per_ex= []
    for b in range(2):
        labels= batch["labels"][b, 1:]
        keep= labels != IGNORE
        per_ex.append(-logp[b][keep].gather(1, labels[keep].unsqueeze(1)).mean())

    assert torch.allclose(loss, torch.stack(per_ex).mean(), atol=1e-5)

    #padding a row does not change its own nll
    alone= example_nll(llm, conn, collate(exs[1:], tok.pad_token_id, left_pad=False), feats[1:])
    assert torch.allclose(example_nll(llm, conn, batch, feats)[1], alone[0], atol=1e-5)


def test_backward_reaches_connector_and_never_the_frozen_backbone(tok):
    llm= tiny_llm(tok)
    conn= Connector(CFG["vision_dim"], TINY_DIM, TINY_DIM)

    batch= collate([make(tok, "How many cats?", "3", True)], tok.pad_token_id, left_pad=False)
    vlm_loss(llm, conn, batch, torch.randn(1, CFG["n_patches"], CFG["vision_dim"])).backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in conn.parameters())
    assert all(p.grad is None for p in llm.parameters())


def test_connector_calibration_matches_text_norm():
    conn= Connector(CFG["vision_dim"], CFG["connector_hidden"], CFG["llm_dim"])
    text= torch.randn(50, CFG["llm_dim"]) * 0.03

    info= calibrate_scale(conn, torch.randn(4, CFG["n_patches"], CFG["vision_dim"]), text)

    out= conn(torch.randn(4, CFG["n_patches"], CFG["vision_dim"]))
    assert out.norm(dim=-1).mean().item() == pytest.approx(info["text_norm"], rel=0.05)
    assert conn.scale.requires_grad is False


def test_vqa_soft_accuracy_matches_the_leave_one_out_formula():
    others= ["cat"] * 10

    #k matching answers out of 10: (k * min(1, (k-1)/3) + (10-k) * min(1, k/3)) / 10
    assert soft_accuracy("dog", ["dog"] * 4 + others[:6]) == pytest.approx(1.0)
    assert soft_accuracy("dog", ["dog"] * 3 + others[:7]) == pytest.approx(0.9)
    assert soft_accuracy("dog", ["dog"] * 2 + others[:8]) == pytest.approx(0.6)
    assert soft_accuracy("dog", ["dog"] * 1 + others[:9]) == pytest.approx(0.3)
    assert soft_accuracy("dog", others) == 0.0

    #normalization: number words, articles, punctuation, case
    assert normalize_answer("Two.") == "2"
    assert normalize_answer("A Dog!") == "dog"
    assert soft_accuracy("two", ["2"] * 10) == pytest.approx(1.0)


def test_pair_bootstrap_resamples_whole_pairs_with_matched_ids():
    #per-pair values for two conditions (e.g. pair accuracy), aligned by pair id
    correct= np.array([1.0, 0.5, 1.0, 0.0, 1.0])
    swapped= np.array([0.0, 0.5, 1.0, 0.0, 0.5])

    idx= resample_indices(len(correct), 1000, seed=6304)
    assert idx.shape == (1000, 5)

    ci= paired_difference_ci(correct, swapped, idx)

    #same indices for both conditions: the interval is the one of the per-resample mean differences
    diffs= (correct[idx] - swapped[idx]).mean(axis=1)
    assert ci["lo"] == pytest.approx(np.percentile(diffs, 2.5))
    assert ci["hi"] == pytest.approx(np.percentile(diffs, 97.5))
    assert ci["mean"] == pytest.approx(0.3)

    #same seed -> same pairs drawn
    assert np.array_equal(idx, resample_indices(5, 1000, seed=6304))


def strict_index():
    #category 1 = dog. image 1: 2 dogs; 2: one crowd dog only; 3: no dog; 4: 1 dog + a crowd dog; 5: 3 dogs
    counts= {1: {1: [2, 0]}, 2: {1: [0, 1]}, 3: {}, 4: {1: [1, 1]}, 5: {1: [3, 0]}}

    return {"counts": counts, "images": [1, 2, 3, 4, 5], "names": {1: "dog"}, "captions": {}}


def test_strict_rule_for_crowd_annotations():
    data_cfg= load_yaml("configs/task2/data.yaml")
    index= strict_index()

    pres= grounding_pairs(index, index["images"], "presence", 200, np.random.default_rng(0), None, data_cfg, "dev")
    yes= {p["items"][0]["image_id"] for p in pres}
    no= {p["items"][1]["image_id"] for p in pres}

    #yes needs a non-crowd dog; no needs zero dog boxes (the crowd-only image 2 is neither)
    assert yes <= {1, 4, 5} and no == {3}
    assert pres[0]["items"][0]["question"] == "Does this image contain a dog?"

    cnt= grounding_pairs(index, index["images"], "counting", 200, np.random.default_rng(0), None, data_cfg, "dev")
    used= {(it["image_id"], it["target"]) for p in cnt for it in p["items"]}

    #image 4 has a crowd dog -> never counted; counts are the non-crowd boxes
    assert used <= {(1, "2"), (5, "3")} and cnt[0]["items"][0]["question"] == "How many dogs can you see?"


def test_isolation_check_rejects_shared_eval_images():
    item= lambda img, ans: {"image_id": img, "split": "val", "question": "q?", "target": ans}
    out= {
        "captions_train": [], "vqa_train": [], "presence_pool_train": [], "counting_pool_train": [],
        "captions_dev": [{"image_id": 7, "split": "val", "caption": "x"}],
        "vqa_dev": [], "vqa_test": [],
        "grounding_dev": [{"pair_id": "a", "items": [item(1, "yes"), item(2, "no")]}],
        "grounding_test": [{"pair_id": "b", "items": [item(2, "yes"), item(3, "no")]}]
    }

    with pytest.raises(AssertionError):
        check_isolation(out)


def test_siglip_last_hidden_state_has_post_layernorm_exactly_once():
    config= SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=16)
    torch.manual_seed(0)
    model= SiglipVisionModel(config).eval()

    #non-trivial affine params: with the default (1, 0) init a layer norm applied twice equals applying it once
    with torch.no_grad():
        model.vision_model.post_layernorm.weight.uniform_(0.5, 2.0)
        model.vision_model.post_layernorm.bias.uniform_(-1.0, 1.0)

    with torch.no_grad():
        out= model(pixel_values=torch.randn(1, 3, 224, 224), output_hidden_states=True)
        normed= model.vision_model.post_layernorm(out.hidden_states[-1])

    #196 patches in 14 x 14 order, already normalized: features.py must not apply the layer norm again
    assert out.last_hidden_state.shape == (1, 196, 32)
    assert torch.allclose(out.last_hidden_state, normed, atol=1e-6)
    assert not torch.allclose(out.last_hidden_state, model.vision_model.post_layernorm(normed), atol=1e-3)


def test_left_padded_generation_matches_unpadded():
    tok= AutoTokenizer.from_pretrained(CFG["llm_name"])
    llm= tiny_llm(tok).eval()
    llm.config.use_cache= True
    conn= Connector(CFG["vision_dim"], TINY_DIM, TINY_DIM).eval()

    cfg= dict(CFG, eval_batch_size=8, eval_max_new_tokens=6)
    segments= prompt_segments(tok, cfg)
    feats= {"val": {"index": {1: 0, 2: 1}, "feats": torch.randn(2, CFG["n_patches"], CFG["vision_dim"]).half()}}

    short= {"image": ("val", 1), "question": "Is it red?", "target": None, "max_target": 16}
    long= {"image": ("val", 2), "question": "How many dogs are sitting on the long wooden bench in the park today?", "target": None, "max_target": 16}

    alone= generate(llm, conn, tok, segments, cfg, [short], feats, torch.device("cpu"))
    padded= generate(llm, conn, tok, segments, cfg, [long, short], feats, torch.device("cpu"))

    assert padded[1]["prediction"] == alone[0]["prediction"]


def test_mismatch_map_never_uses_the_pairs_own_images():
    pairs= [{"pair_id": f"p{i}"} for i in range(7)]
    m= mismatch_map(pairs, 6304)

    assert all(m[p] != p for p in m) and sorted(m.values()) == sorted(m)
    assert m == mismatch_map(pairs, 6304)
