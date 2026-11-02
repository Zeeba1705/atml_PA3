import numpy as np

from task3.answers import NO_ANSWER
from task3.selectors import rwc, rwc_margin, self_consistency, vote

#handout p. 13: stopping is checked after these many revealed candidates, otherwise all 32 are used
CHECKPOINTS= [4, 8, 16]


def agreement_stop(answers, tau):
    #unique, non-NO_ANSWER leading answer supported by at least tau * n of the n revealed candidates
    _, totals= vote(answers, np.ones(len(answers)))
    counts= sorted(totals.items(), key=lambda kv: -kv[1])

    leader, top= counts[0]
    unique= len(counts) == 1 or counts[1][1] < top

    return unique and leader != NO_ANSWER and top >= tau * len(answers)


def quality_stop(answers, scores, tau, gamma):
    #agreement AND self-consistency winner == rwc winner AND rwc margin >= gamma (scores from revealed only)
    if not agreement_stop(answers, tau):
        return False

    return self_consistency(answers) == rwc(answers, scores) and rwc_margin(answers, scores) >= gamma
