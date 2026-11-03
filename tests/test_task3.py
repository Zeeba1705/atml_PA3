import numpy as np
import pytest

from task3.adaptive import agreement_stop, quality_stop
from task3.answers import NO_ANSWER, is_correct
from task3.scores import calibrate, candidate_scores, visconf
from task3.selectors import average_ranks, highest_score, rwc, rwc_margin, rwc_weights, self_consistency, vote


def test_rwc_weights_with_average_rank_ties():
    #scores 0.9, 0.5, 0.5, 0.1 -> ranks 1, 2.5, 2.5, 4 -> w = (4 - r) / 3
    scores= np.array([0.9, 0.5, 0.5, 0.1])

    assert average_ranks(scores).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert np.allclose(rwc_weights(scores), [1.0, 0.5, 0.5, 0.0])

    #one candidate gets the whole vote
    assert rwc_weights(np.array([0.3])).tolist() == [1.0]


def test_rwc_vote_uses_rank_weights():
    #"b" has two mid-ranked rollouts (0.5 + 0.5) and ties "a" (1.0); "a" supports first in the order
    answers= ["a", "b", "b", "c"]
    scores= np.array([0.9, 0.5, 0.5, 0.1])

    assert rwc(answers, scores) == "a"

    #same weights, but "b" now appears first -> "b"
    assert rwc(["b", "a", "b", "c"], np.array([0.5, 0.9, 0.5, 0.1])) == "b"


def test_ties_go_to_the_earliest_supporting_rollout():
    #self-consistency 2-2 tie: "y" is supported earlier than "x"
    assert self_consistency(["y", "x", "x", "y"]) == "y"
    assert self_consistency(["x", "y", "x", "y"]) == "x"

    #highest score tie: earliest rollout wins
    assert highest_score(["p", "q", "r"], np.array([0.2, 0.7, 0.7])) == "q"


def test_visconf_calibration_uses_only_the_revealed_prefix():
    rng= np.random.default_rng(0)
    pool= {
        "logp": [-rng.random(rng.integers(5, 20)) for _ in range(33)],
        "kl": [rng.random(rng.integers(5, 20)) * 5 for _ in range(33)],
        "v": [rng.random(1) for _ in range(33)]
    }
    for i in range(33):
        pool["v"][i]= rng.random(len(pool["kl"][i]))

    order= list(rng.permutation(32))
    prefix= order[:8]

    before= candidate_scores(pool, prefix, "visconf")

    #a 33rd rollout with extreme KL joins the pool but is not revealed: prefix scores must not move
    pool["kl"][32]= np.full(10, 1e6)
    pool["v"][32]= np.full(10, 0.5)
    assert np.array_equal(candidate_scores(pool, prefix, "visconf"), before)

    #revealing it does change the calibration (the check above is not vacuous)
    assert not np.allclose(candidate_scores(pool, prefix + [32], "visconf")[:8], before)


def test_tokenwise_fusion_minus_visconf_is_token_covariance():
    rng= np.random.default_rng(1)
    kl= [rng.random(n) * 3 for n in [7, 12, 4]]
    v= [rng.random(len(k)) for k in kl]

    Z, Z_tok, _= visconf(kl, v)
    c= calibrate(kl)

    #population covariance over tokens: mean(cv) - mean(c) mean(v)
    cov= np.array([np.mean(ci * vi) - np.mean(ci) * np.mean(vi) for ci, vi in zip(c, v)])
    assert np.allclose(Z_tok - Z, cov)


def test_calibration_clips_to_unit_interval_with_linear_percentiles():
    #pooled 0..100: Q05 = 5, Q95 = 95 with linear interpolation
    c= calibrate([np.arange(0, 51, dtype=float), np.arange(51, 101, dtype=float)])

    assert c[0][0] == 0.0 and c[1][-1] == 1.0
    assert c[0][50] == pytest.approx((50 - 5) / 90)


def test_no_answer_is_never_correct_and_still_votes():
    assert not is_correct(NO_ANSWER, NO_ANSWER)
    assert is_correct("4", "4")

    #three NO_ANSWER rollouts outvote two real answers: self-consistency returns NO_ANSWER (graded wrong)
    answers= [NO_ANSWER, "4", NO_ANSWER, "5", NO_ANSWER]
    assert self_consistency(answers) == NO_ANSWER
    assert vote(answers, np.ones(5))[1][NO_ANSWER] == 3.0

    #agreement stopping never stops on a NO_ANSWER leader
    assert not agreement_stop([NO_ANSWER] * 4, tau=0.5)


def test_adaptive_margin_is_one_with_a_single_answer_group():
    answers= ["7", "7", "7", "7"]
    scores= np.array([0.1, 0.4, 0.3, 0.2])

    assert rwc_margin(answers, scores) == 1.0
    assert quality_stop(answers, scores, tau=1.0, gamma=0.5)


def test_agreement_needs_a_unique_leader_with_enough_support():
    assert agreement_stop(["a", "a", "a", "b"], tau=0.75)
    assert not agreement_stop(["a", "a", "a", "b"], tau=0.875)

    #2-2 tie: no unique leader
    assert not agreement_stop(["a", "b", "a", "b"], tau=0.5)


def test_math_groups_equivalent_answers_without_gold():
    from task3.answers import grade_math, group_math

    keys, reps= group_math(["\\frac{1}{2}", "0.5", None, "2", "  "])

    #0.5 joins the \frac{1}{2} group; missing / blank extractions are NO_ANSWER
    assert keys == ["\\frac{1}{2}", "\\frac{1}{2}", NO_ANSWER, "2", NO_ANSWER]
    assert grade_math(reps, ".5") == "\\frac{1}{2}"
    assert grade_math(reps, "7") is None


def test_mathvista_unit_stripping_and_choice_rules():
    from task3.answers import mathvista_answer, mathvista_gold

    integer= {"question_type": "free_form", "answer_type": "integer", "precision": None, "answer": "1000"}
    flt= {"question_type": "free_form", "answer_type": "float", "precision": "2", "answer": "0.33"}
    mc= {"question_type": "multi_choice", "answer_type": "text", "choices": ["135°", "140°", "145°"], "answer": "140°"}

    assert mathvista_answer("1000 \\text{ ml}", integer) == "1000"
    assert mathvista_answer("1,200", integer) == "1200"
    assert mathvista_answer("10-20 years", integer) == NO_ANSWER
    assert mathvista_answer("0.3333 \\, \\text{m}", flt) == "0.33"

    assert mathvista_gold(mc) == "B"
    assert mathvista_answer("(B)", mc) == "B"
    assert mathvista_answer("145°", mc) == "C"
    assert mathvista_answer(None, mc) == NO_ANSWER
