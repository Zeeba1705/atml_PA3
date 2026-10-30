import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from task1.data import build_example, collate, normalize_question, select_examples
from task1.diagnostics import activation_ratio, eta, rho_k
import task1.losses as losses
from task1.losses import accumulate_sft_grads, ce_sum, count_tokens
from task1.merge import merge_adapters
import task1.methods.l2sp as l2sp
from task1.methods.l2sp import add_l2sp_grad, l2sp_omega, snapshot_params, unique_params
from task1.methods.lora import LoRALinear, add_lora
from task1.methods.oplora import OPLoRALinear, add_oplora, oplora_constraint_check, project_factors, svd_bases

MODEL_NAME= "Qwen/Qwen3-0.6B"
TARGETS= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_qwen3(seed=0):
    config= Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True
    )

    torch.manual_seed(seed)
    model= Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa", dtype=torch.float32)

    return model


def toy_microbatches():
    #4 examples of different lengths and prompt lengths, split into 2 microbatches of 2
    gen= torch.Generator().manual_seed(1)

    examples= []
    for length, n_prompt in [(9, 4), (6, 2), (12, 7), (5, 3)]:
        ids= torch.randint(0, 64, (length,), generator=gen)
        labels= ids.clone()
        labels[:n_prompt]= -100
        examples.append({"input_ids": ids, "labels": labels})

    return [collate(examples[0:2], pad_id=0), collate(examples[2:4], pad_id=0)], examples


#--- data ---

def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        pytest.skip(f"cannot load {MODEL_NAME} tokenizer: {e}")


def test_label_mask_decodes_to_response_and_im_end():
    tok= load_tokenizer()

    question= "Janet has 3 apples and buys 2 more. How many?"
    response= "Janet starts with 3 apples.\nShe buys $2$ more, so $3+2=5$.\nThe answer is: 5"

    ex= build_example(tok, question, response)

    ids= ex["input_ids"]
    labels= ex["labels"]

    assert ids.shape == labels.shape

    supervised= ids[labels != -100]
    assert tok.decode(supervised) == response + "<|im_end|>"

    #last supervised token is the end-of-turn token, nothing after it
    assert ids[-1].item() == tok.convert_tokens_to_ids("<|im_end|>")
    assert labels[-1].item() == ids[-1].item()

    #everything masked is exactly the non-thinking generation prompt (incl. the empty think block)
    prompt= tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    assert "<think>\n\n</think>\n\n" in prompt
    assert tok.decode(ids[labels == -100]) == prompt

    #masked tokens form one prefix block
    n_prompt= int((labels == -100).sum())
    assert (labels[:n_prompt] == -100).all()
    assert (labels[n_prompt:] != -100).all()

    assert ex["length"] == len(ids)


def test_label_mask_response_starting_with_newline():
    #~2% of MetaMathQA responses start with "\n"; it must not merge into the prompt's last token
    tok= load_tokenizer()

    question= "Set x = 30. What is x?"
    response= "\nSetting $x = 30$ gives 30.\nThe answer is: 30"

    ex= build_example(tok, question, response)

    prompt_ids= tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    prompt_ids= tok(prompt_ids, add_special_tokens=False)["input_ids"]

    n_prompt= int((ex["labels"] == -100).sum())

    assert ex["input_ids"][:n_prompt].tolist() == prompt_ids
    assert tok.decode(ex["input_ids"][ex["labels"] != -100]) == response + "<|im_end|>"


def test_normalize_question():
    assert normalize_question("  Janet’s ducks lay 16 eggs\nper DAY. ") == "janet s ducks lay 16 eggs per day"
    assert normalize_question("What is $2+3$?") == normalize_question("what is 2 + 3")
    assert normalize_question("ｆｕｌｌ width") == "full width"   #nfkc


def test_select_examples_order_reasons_and_length_limit():
    tok= load_tokenizer()

    gsm8k= ["Janet’s ducks lay 16 eggs per day. How many?"]

    records= [
        {"query": "Q0 what is 1+1?", "response": "2", "original_question": "x"},
        {"query": "   ", "response": "empty question", "original_question": "x"},
        {"query": "janet's ducks lay 16 eggs per day how many", "response": "16", "original_question": "y"},
        {"query": "rephrased", "response": "r", "original_question": "Janet’s ducks lay 16 eggs per day. How many?"},
        {"query": "Q4 long", "response": "word " * 400, "original_question": "z"},
        {"query": "Q5 what is 2+2?", "response": "4", "original_question": "z"},
        {"query": "Q6 never reached", "response": "6", "original_question": "z"}
    ]

    #a prompt+response of a few tokens fits, the 400-word response does not
    kept, examples, exclusions= select_examples(records, tok, gsm8k, n_examples=2, max_length=100)

    assert kept == [0, 5]
    assert exclusions == {"empty": [1], "gsm8k_overlap": [2, 3], "too_long": [4]}
    assert [tok.decode(ex["input_ids"][ex["labels"] != -100]) for ex in examples] == ["2<|im_end|>", "4<|im_end|>"]

    #length exactly max_length is kept
    exact= examples[0]["length"]
    kept, _, exclusions= select_examples(records[:1], tok, gsm8k, n_examples=1, max_length=exact)
    assert kept == [0]

    with pytest.raises(ValueError):
        select_examples(records[:1], tok, gsm8k, n_examples=1, max_length=exact - 1)


def test_collate_pads_right_with_masked_labels():
    a= {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7])}
    b= {"input_ids": torch.tensor([8]), "labels": torch.tensor([8])}

    batch= collate([a, b], pad_id=0)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


#--- token-normalized loss ---

def test_token_normalized_loss_hand_computed():
    ln2= math.log(2.0)
    ln3= math.log(3.0)

    #microbatch 1: shifted targets are [2, 0] -> 2 valid tokens
    logits1= torch.tensor([[[0.0, 0.0, 0.0], [ln2, 0.0, 0.0], [9.0, 9.0, 9.0]]])
    labels1= torch.tensor([[-100, 2, 0]])

    #microbatch 2: shifted targets are [-100, 1, -100] -> 1 valid token
    logits2= torch.tensor([[[5.0, 1.0, 2.0], [0.0, ln3, 0.0], [1.0, 1.0, 1.0], [7.0, 0.0, 0.0]]])
    labels2= torch.tensor([[-100, -100, 1, -100]])

    #by hand: -log(1/3), -log(2/4), -log(3/5)
    nll= [math.log(3.0), math.log(2.0), math.log(5.0 / 3.0)]

    assert count_tokens(labels1) == 2
    assert count_tokens(labels2) == 1

    s1= ce_sum(logits1, labels1)
    s2= ce_sum(logits2, labels2)

    assert s1.item() == pytest.approx(nll[0] + nll[1], abs=1e-6)
    assert s2.item() == pytest.approx(nll[2], abs=1e-6)

    total= (s1 + s2) / (count_tokens(labels1) + count_tokens(labels2))
    assert total.item() == pytest.approx(sum(nll) / 3, abs=1e-6)

    #a mean of microbatch means would give a different number here
    mean_of_means= ((nll[0] + nll[1]) / 2 + nll[2]) / 2
    assert abs(total.item() - mean_of_means) > 1e-3


@pytest.mark.parametrize("ce_chunk", [1024, 3])
def test_accumulated_supervised_loss_matches_full_logits_big_batch(ce_chunk, monkeypatch):
    #ce_chunk=3 splits the lm head / CE into many checkpointed chunks
    monkeypatch.setattr(losses, "CE_CHUNK", ce_chunk)

    microbatches, examples= toy_microbatches()

    #reference: one padded batch of all 4 examples, full logits, sum CE / all valid tokens
    ref_model= tiny_qwen3()
    big= collate(examples, pad_id=0)

    logits= ref_model(input_ids=big["input_ids"], attention_mask=big["attention_mask"]).logits
    n_tok= count_tokens(big["labels"])

    ref_loss= ce_sum(logits, big["labels"]) / n_tok
    ref_loss.backward()

    #ours: 2 microbatches, lm head only at supervised positions, normalized by total tokens of the update
    model= tiny_qwen3()

    loss, n_tokens= accumulate_sft_grads(
        model=model,
        microbatches=microbatches,
        scaler=None,
        device=torch.device("cpu"),
        use_fp16=False
    )

    assert n_tokens == n_tok
    assert loss.item() == pytest.approx(ref_loss.item(), abs=1e-6)

    ref_params= dict(ref_model.named_parameters())
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, ref_params[name].grad, atol=1e-6), name


#--- L2-SP ---

class TiedToy(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb= nn.Embedding(5, 3)
        self.norm= nn.Parameter(torch.ones(3))
        self.head= nn.Linear(3, 5, bias=True)

        #tied like Qwen's embed_tokens / lm_head
        self.head.weight= self.emb.weight

    def forward(self, ids):
        return self.head(self.emb(ids) * self.norm)


@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_l2sp_omega_counts_tied_weight_once(chunk, monkeypatch):
    #chunk sizes 1 and 2 make the foreach loop cross chunk boundaries
    monkeypatch.setattr(l2sp, "CHUNK", chunk)

    model= TiedToy()

    params= unique_params(model)
    assert len(params) == 3   #emb/head weight once, norm, head bias

    theta0= snapshot_params(params)

    with torch.no_grad():
        model.emb.weight.add_(1.0)   #15 entries, also moves head.weight
        model.head.bias.add_(2.0)    #5 entries
        model.norm.add_(3.0)         #3 entries

    #0.5 * (15*1 + 5*4 + 3*9) = 31, counting the tied matrix twice would give 38.5
    omega= l2sp_omega(params, theta0)
    assert omega.item() == pytest.approx(31.0, abs=1e-5)

    #penalty gradient added to .grad equals autograd of lambda*omega
    lam= 1e-3

    for p in params:
        p.grad= torch.zeros_like(p)

    add_l2sp_grad(params, theta0, lam)

    auto_params= [p.detach().clone().requires_grad_(True) for p in params]
    auto_omega= 0.5 * sum(((p - t) ** 2).sum() for p, t in zip(auto_params, theta0))
    (lam * auto_omega).backward()

    for p, q in zip(params, auto_params):
        assert torch.allclose(p.grad, q.grad, atol=1e-8)


#--- OPLoRA ---

def test_oplora_projections_are_orthogonal_to_protected_bases():
    torch.manual_seed(0)

    W0= torch.randn(40, 30)
    A= torch.randn(4, 30)
    B= torch.randn(40, 4)

    U_k, V_k= svd_bases(W0, k=8)

    assert U_k.shape == (40, 8)
    assert V_k.shape == (30, 8)

    B_t, A_t= project_factors(A, B, U_k, V_k)

    assert torch.linalg.norm(U_k.T @ B_t) < 1e-5
    assert torch.linalg.norm(A_t @ V_k) < 1e-5

    #so the update leaves the protected input and output subspaces alone
    delta= B_t @ A_t
    assert torch.linalg.norm(U_k.T @ delta) < 1e-4
    assert torch.linalg.norm(delta @ V_k) < 1e-4

    #projection removes something from random factors
    assert torch.linalg.norm(B - B_t) > 1e-2
    assert torch.linalg.norm(A - A_t) > 1e-2


def test_oplora_initial_factors_match_lora16():
    lora_model= tiny_qwen3()
    oplora_model= tiny_qwen3()

    add_lora(lora_model, TARGETS, rank=16, alpha=32, dropout=0.05, seed=6304)
    add_oplora(oplora_model, TARGETS, rank=16, alpha=32, dropout=0.05, seed=6304, k=4)

    lora_layers= {n: m for n, m in lora_model.named_modules() if isinstance(m, LoRALinear)}
    oplora_layers= {n: m for n, m in oplora_model.named_modules() if isinstance(m, LoRALinear)}

    assert len(lora_layers) == 2 * 7
    assert lora_layers.keys() == oplora_layers.keys()

    for name in lora_layers:
        assert torch.equal(lora_layers[name].lora_A, oplora_layers[name].lora_A), name
        assert torch.equal(lora_layers[name].lora_B, oplora_layers[name].lora_B), name

    #zero B means the adapters start as the identity map
    assert all((m.lora_B == 0).all() for m in lora_layers.values())


#--- diagnostics ---

def test_rho_eta_and_activation_ratio_hand_computed():
    W0= torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    U_k, _= svd_bases(W0, k=1)   #U_k = +-e1

    delta= torch.tensor([
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    #first row energy 5 out of total 9
    assert rho_k(delta, U_k).item() == pytest.approx(5.0 / 9.0, abs=1e-6)

    #||delta||_F = 3, ||W0||_F = sqrt(14)
    assert eta(delta, W0).item() == pytest.approx(3.0 / math.sqrt(14.0), abs=1e-6)

    #z = e1: delta z = (1,0,0), W0 z = (3,0,0) -> 1/3 ; z = e3: (0,0,2) vs (0,0,1) -> 2
    z= torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r= activation_ratio(delta, W0, z)
    assert torch.allclose(r, torch.tensor([1.0 / 3.0, 2.0]), atol=1e-6)


#--- merging ---

@pytest.mark.parametrize("method", ["lora", "oplora"])
def test_merged_vs_unmerged_logits_match(method):
    model= tiny_qwen3()

    if method == "lora":
        add_lora(model, TARGETS, rank=4, alpha=8, dropout=0.05, seed=6304)
    else:
        add_oplora(model, TARGETS, rank=4, alpha=8, dropout=0.05, seed=6304, k=4)

    #B starts at zero, give it values so the merge actually changes weights
    torch.manual_seed(3)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)

    model.eval()

    ids= torch.randint(0, 64, (2, 10))
    mask= torch.ones_like(ids)

    with torch.no_grad():
        unmerged= model(input_ids=ids, attention_mask=mask).logits
        base_logits= tiny_qwen3()(input_ids=ids, attention_mask=mask).logits

    merged_model= merge_adapters(copy.deepcopy(model))

    assert not any(isinstance(m, LoRALinear) for m in merged_model.modules())

    with torch.no_grad():
        merged= merged_model(input_ids=ids, attention_mask=mask).logits

    assert (merged - unmerged).abs().max().item() < 1e-5
    assert (merged - base_logits).abs().max().item() > 1e-3


def test_oplora_projection_stays_fp32_under_autocast():
    model= tiny_qwen3()
    add_oplora(model, TARGETS, rank=4, alpha=8, dropout=0.0, seed=6304, k=4)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)

    layer= next(m for m in model.modules() if isinstance(m, OPLoRALinear))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        A_t, B_t= layer.factors()

    assert A_t.dtype == torch.float32 and B_t.dtype == torch.float32

    worst_B, worst_A= oplora_constraint_check(model)
    assert worst_B < 1e-5 and worst_A < 1e-5


def test_theta0_placement():
    from task1.train import theta0_fits_on_gpu

    #full_ft's measured peak on a T4 + 2.4 GB theta0 is over 93%: stays on the cpu
    assert not theta0_fits_on_gpu(12.55e9, 2.4e9, 15.64e9, 0.93)
    assert theta0_fits_on_gpu(8e9, 2.4e9, 15.64e9, 0.93)

    #the cpu snapshot is an exact fp32 copy, not tied to the parameters
    params= [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    theta0= snapshot_params(params, torch.device("cpu"))
    assert all(torch.equal(t, p.detach()) and t.data_ptr() != p.data_ptr() for t, p in zip(theta0, params))
