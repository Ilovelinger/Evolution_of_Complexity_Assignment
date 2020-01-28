import matplotlib.pyplot as plt
from Rastrigin import normalGA1
from Rastrigin import CCGA1
from Schwefel import normalGA2
from Schwefel import CCGA2
from Griewangk import normalGA3
from Griewangk import CCGA3
from Ackley import normalGA4
from Ackley import CCGA4


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def show1(iteration):
    n1 = 20
    n2 = 10
    n3 = 10
    n4 = 30
    List1 = zerolistmaker(iteration)
    for i in range(5):
        Y1 = normalGA1(n1, iteration)
        List1 = [i + j for i, j in zip(List1, Y1)]

    for i in range(len(List1)):
        List1[i] = List1[i] / 5

    List2 = zerolistmaker(iteration)
    for i in range(5):
        Y2 = CCGA1(n1, iteration)
        List2 = [i + j for i, j in zip(List2, Y2)]
    for i in range(len(List2)):
        List2[i] = List2[i] / 5

    X = []
    for i in range(iteration):
        X.append(i)
    plt.xlabel("Function evaluation")
    plt.ylabel("Best individual")
    plt.title("Rastrigin Function")
    plt.plot(X, List1, c='k', linestyle='-')
    plt.plot(X, List2, c='k', linestyle='--')
    plt.legend(["Standard GA", "CCGA-1"])
    plt.show()

    # Y1 = normalGA2(n2, iteration)
    # Y2 = CCGA2(n2, iteration)
    # X = []
    # for i in range(iteration):
    #     X.append(i)
    # plt.xlabel("Function evaluation")
    # plt.ylabel("Best individual")
    # plt.title("Schwefel Function")
    # plt.plot(X, Y1, c='k', linestyle='-')
    # plt.plot(X, Y2, c='k', linestyle='--')
    # plt.legend(["Standard GA", "CCGA-1"])
    # plt.show()
    #
    # Y1 = normalGA3(n3, iteration)
    # Y2 = CCGA3(n3, iteration)
    # X = []
    # for i in range(iteration):
    #     X.append(i)
    # plt.xlabel("Function evaluation")
    # plt.ylabel("Best individual")
    # plt.title("Griewangk Function")
    # plt.plot(X, Y1, c='k', linestyle='-')
    # plt.plot(X, Y2, c='k', linestyle='--')
    # plt.legend(["Standard GA", "CCGA-1"])
    # plt.show()
    #
    # Y1 = normalGA4(n4, iteration)
    # Y2 = CCGA4(n4, iteration)
    # X = []
    # for i in range(iteration):
    #     X.append(i)
    # plt.xlabel("Function evaluation")
    # plt.ylabel("Best individual")
    # plt.title("Ackley Function")
    # plt.plot(X, Y1, c='k', linestyle='-')
    # plt.plot(X, Y2, c='k', linestyle='--')
    # plt.legend(["Standard GA", "CCGA-1"])
    # plt.show()

# show1(100)
