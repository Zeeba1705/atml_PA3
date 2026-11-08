import numpy as np

#handout p. 12: c = clip((k - Q05) / max(Q95 - Q05, 1e-8)) with linear-interpolation percentiles
CALIB_EPS= 1e-8
CALIB_QUANTILES= [5, 95]


def mean_logprob(logp):
    #L_i = (1/T_i) sum_t log p_i,t(y_i,t), one float per rollout (logp: list of per-token arrays)
    return np.array([np.mean(x) if len(x) else np.nan for x in logp])


def self_certainty(kl):
    #K_i = (1/T_i) sum_t KL(U || p_i,t)
    return np.array([np.mean(x) if len(x) else np.nan for x in kl])


def vision_share(a_img, a_prompt, a_gen):
    #v_i,t = A_img / (A_img + A_prompt + A_gen), per token of one rollout (the totals need not sum to one)
    a_img= np.asarray(a_img, dtype=np.float64)
    tot= a_img + np.asarray(a_prompt, dtype=np.float64) + np.asarray(a_gen, dtype=np.float64)

    return np.divide(a_img, tot, out=np.zeros_like(a_img), where=tot > 0)


def vision_only(v):
    #V_i = mean_t v_i,t
    return np.array([np.mean(x) if len(x) else np.nan for x in v])


def calibrate(kl):
    #pool token KL over the CURRENT candidate set only, so unseen rollouts never enter the calibration
    pooled= np.concatenate([np.asarray(x, dtype=np.float64) for x in kl])
    q05, q95= np.percentile(pooled, CALIB_QUANTILES)

    scale= max(q95 - q05, CALIB_EPS)

    return [np.clip((np.asarray(x, dtype=np.float64) - q05) / scale, 0.0, 1.0) for x in kl]


def visconf(kl, v):
    #Z_i = V_i C_i (rollout summaries multiplied) and the tokenwise ablation Z_tok_i = mean_t c_i,t v_i,t
    c= calibrate(kl)

    C= np.array([np.mean(x) if len(x) else np.nan for x in c])
    V= vision_only(v)
    Z_tok= np.array([np.mean(ci * vi) if len(ci) else np.nan for ci, vi in zip(c, v)])

    return V * C, Z_tok, C


def candidate_scores(pool, ids, score):
    #quality scores of the revealed candidates `ids` (current order), computed from those candidates only.
    #pool: one question's per-rollout token arrays, indexed by rollout id ("logp", "kl", and for the vlm "v")
    logp= [pool["logp"][i] for i in ids]
    kl= [pool["kl"][i] for i in ids]

    if score == "logprob":
        return mean_logprob(logp)
    if score == "self_certainty":
        return self_certainty(kl)

    v= [pool["v"][i] for i in ids]
    if score == "vision":
        return vision_only(v)

    Z, Z_tok, _= visconf(kl, v)
    if score == "visconf":
        return Z
    if score == "visconf_token":
        return Z_tok

    raise ValueError(f"unknown score {score}")
