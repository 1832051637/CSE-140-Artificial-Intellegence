"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None


def question2():
    """
    [Enter a description of what you did here.]
    Here I change a smaller noise, so that the agent will be less easier to 
    acciently fall off from the bridge
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise


def question3a():
    """
    [Enter a description of what you did here.]
    If the livingReward is negative but not large penalty, then 
    the agent would prefer to end quick at 1 instead of 10
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -4.0

    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    """
    [Enter a description of what you did here.]
    Large noise makes the agent avoid cliff, but 
    negative reward and smaller discount makes agent prefer nearer exit
    """

    answerDiscount = 0.7
    answerNoise = 0.5
    answerLivingReward = -1.2

    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    """
    [Enter a description of what you did here.]
    Smaller noise makes the agent risk the cliff
    """

    answerDiscount = 0.9
    answerNoise = 0.05
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    """
    [Enter a description of what you did here.]
    Large noise but no living penalty makes the agent
    avoid cliff and exit at +10.
    """

    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    """
    [Enter a description of what you did here.]
    Positive living reward makes agent to stay in maze as long as possible
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1.0

    return answerDiscount, answerNoise, answerLivingReward


def question6():
    """
    [Enter a description of what you did here.]
    It seems it's impossible to get a correct policy in 50 iteration
    """
    return NOT_POSSIBLE
    # answerEpsilon = 0.3
    # answerLearningRate = 0.5

    # return answerEpsilon, answerLearningRate


if __name__ == "__main__":
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print("Answers to analysis questions:")
    for question in questions:
        response = question()
        print("    Question %-10s:\t%s" % (question.__name__, str(response)))
