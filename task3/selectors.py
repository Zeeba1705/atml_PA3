import numpy as np

#all functions take the current candidates in their revealed order: index 0 = first revealed rollout.
#answers are hashable group keys (NO_ANSWER is an ordinary group here: it votes, and grading marks it wrong)


def clean_scores(scores):
    #a rollout with no scored tokens gets the lowest possible score instead of breaking the ranking
    scores= np.asarray(scores, dtype=np.float64)

    return np.where(np.isnan(scores), -np.inf, scores)


def average_ranks(scores):
    #descending ranks, 1 = best; tied scores share the mean of the ranks they span
    scores= clean_scores(scores)
    order= np.argsort(-scores, kind="stable")

    ranks= np.empty(len(scores))
    i= 0
    while i < len(order):
        j= i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j+= 1

        ranks[order[i:j + 1]]= (i + j) / 2 + 1
        i= j + 1

    return ranks


def rwc_weights(scores):
    #w_i = (N - r_i) / (N - 1); with one candidate it simply gets the full vote
    n= len(scores)
    if n == 1:
        return np.ones(1)

    return (n - average_ranks(scores)) / (n - 1)


def vote(answers, weights):
    #W(a) = sum of weights of the rollouts answering a. ties -> the answer whose earliest supporting rollout comes
    #first in the current order. returns (winner, {answer: W(a)})
    totals= {}
    first_seen= {}
    for i, (a, w) in enumerate(zip(answers, weights)):
        totals[a]= totals.get(a, 0.0) + float(w)
        first_seen.setdefault(a, i)

    winner= max(totals, key=lambda a: (totals[a], -first_seen[a]))

    return winner, totals


def first_answer(answers):
    return answers[0]


def self_consistency(answers):
    return vote(answers, np.ones(len(answers)))[0]


def highest_score(answers, scores):
    #argmax score; ties -> earliest rollout in the current order (np.argmax returns the first maximum)
    return answers[int(np.argmax(clean_scores(scores)))]


def rwc(answers, scores):
    return vote(answers, rwc_weights(scores))[0]


def rwc_margin(answers, scores):
    #(largest - second largest answer vote) / total vote weight; runner-up = 0 with a single answer group
    _, totals= vote(answers, rwc_weights(scores))
    votes= sorted(totals.values(), reverse=True)

    top= votes[0]
    second= votes[1] if len(votes) > 1 else 0.0
    total= sum(votes)

    return (top - second) / total if total > 0 else 0.0
