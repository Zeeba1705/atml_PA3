#handout p. 12: missing extractions form one NO_ANSWER group, always graded incorrect, but kept in voting and cost
NO_ANSWER= "NO_ANSWER"


def is_correct(group, gold_group):
    #groups come from the same equivalence rules for voting and grading; NO_ANSWER never matches gold
    return group != NO_ANSWER and group == gold_group
