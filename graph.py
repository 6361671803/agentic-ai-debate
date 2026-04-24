import matplotlib.pyplot as plt
import numpy as np

def plot_graph(opt, pes, ana):

    rounds = ["Round 1", "Round 2"]

    x = np.arange(len(rounds))
    width = 0.25

    plt.bar(x - width, opt, width, label="Optimist")
    plt.bar(x, pes, width, label="Pessimist")
    plt.bar(x + width, ana, width, label="Analyst")

    plt.xlabel("Rounds")
    plt.ylabel("Confidence")
    plt.title("Agent Confidence Debate")

    plt.xticks(x, rounds)
    plt.legend()

    plt.show()
