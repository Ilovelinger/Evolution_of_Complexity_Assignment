import math
import random
import string
import struct

import numpy
import numpy as np
from random import seed
from random import randint
import matplotlib.pyplot as plt


def Ackley(individual):
    n = 30
    memberList = []
    i = 0
    while i != len(individual):
        member = individual[i:i + 16].copy()
        i = i + 16
        memberList.append(member)
    decimalnumberList = []
    for j in range(len(memberList)):
        decimalnumber = converter(memberList[j])
        decimalnumberList.append(decimalnumber)
    scalenumberList = []
    for k in range(len(decimalnumberList)):
        scalenumber = -30.0 + ((30.0 + 30.0) / 2 ** 16 * decimalnumberList[k])
        scalenumberList.append(scalenumber)
    firstSum = 0.0
    secondSum = 0.0
    for c in scalenumberList:
        firstSum += c ** 2.0
        secondSum += math.cos(2.0 * math.pi * c)
    result = -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e
    return result


class Member():

    def __init__(self, bitsString=None):
        self.bitsString = bitsString

        self.fitness = Ackley(bitsString)


def createIndividual(n):
    x = numpy.random.randint(2, size=(16 * n,))
    new_individual = Member(x)
    return new_individual


def Mutation(parent, n):
    childbitsString = [0] * 16 * n
    child = Member(childbitsString)
    for i in range(len(parent.bitsString)):
        p = random.uniform(0, 1)
        if p < (1 / len(parent.bitsString)):
            if parent.bitsString[i] == 1:
                child.bitsString[i] = 0
            else:
                child.bitsString[i] = 1
        else:
            child.bitsString[i] = parent.bitsString[i]
    child.fitness = Ackley(child.bitsString)
    return child


def twoPointCrossover(x, y, n):
    parent1 = x.bitsString
    parent2 = y.bitsString
    child1 = parent1.copy()
    child2 = parent2.copy()
    length = len(parent1)
    crossoverPoint1 = random.randrange(16 * n)
    crossoverPoint2 = random.randrange(16 * n)
    while crossoverPoint1 == crossoverPoint2:
        crossoverPoint2 = random.randrange(16 * n)
    points = sorted([crossoverPoint1, crossoverPoint2])
    p1, p2 = points[0], points[1]
    p = random.uniform(0, 1)
    if p < 0.6:
        for i in range(p1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        for j in range(p1, p2):
            child1[j] = parent2[j]
            child2[j] = parent1[j]
        for k in range(p2, length):
            child1[k] = parent1[k]
            child2[k] = parent2[k]
    newchild1 = Member(child1)
    newchild1.fitness = Ackley(child1)
    newchild2 = Member(child2)
    newchild2.fitness = Ackley(child2)
    return newchild1, newchild2


def converter(x):
    length = len(x)
    num = 0
    for i in range(length):
        num = num + int(x[i])
        num = num * 2
    return int(num / 2)


def findBestinList(list):
    index = 0
    for i in range(1, len(list)):
        if list[i].fitness < list[i - 1].fitness:
            index = i
    return index


def findWorstinList(list):
    index = 0
    for i in range(1, len(list)):
        if list[i].fitness > list[i - 1].fitness:
            index = i
    return index


def roulette_selection(population):
    minIndex = findBestinList(population)
    fitnessList = []
    for i in range(len(population)):
        fitnessList.append(population[i].fitness)
    sum_fitness = sum(fitnessList)
    max_fs = max(fitnessList)
    min_fs = min(fitnessList)
    p = random.uniform(0, sum_fitness)
    t = max_fs + min_fs
    choosen = population[0]
    for i in population:
        p -= (t - Ackley(i.bitsString))
        if p < 0:
            if i == population[minIndex]:
                continue
            choosen = i
            break
    return choosen


def compare(a, b):
    length = len(a)
    for i in range(length):
        if b[i] != a[i]:
            return False
    return True


def normalGA4(n, iteration):
    generation = 0
    population = []
    for a in range(100):
        b = createIndividual(n)
        population.append(b)
    currentFitness = []
    for i in range(len(population)):
        individual = population[i]
        fitness = individual.fitness
        currentFitness.append(fitness)
    bestFitnessList1 = []
    while generation < iteration:
        generation = generation + 1
        # print("current generation: ", generation)
        parent1 = roulette_selection(population)
        parent2 = roulette_selection(population)
        while compare(parent1.bitsString, parent2.bitsString):
            parent2 = roulette_selection(population)
        crossoverchild1, crossoverchild2 = twoPointCrossover(parent1, parent2, n)
        mutationchild1 = Mutation(crossoverchild1, n)
        mutationchild2 = Mutation(crossoverchild2, n)
        mutationchildFitnessList = [mutationchild1.fitness, mutationchild2.fitness]
        mutationchildFitnessList.sort()
        bestMutationChildFitness = mutationchildFitnessList[0]
        currentFitness.sort()
        bestFitness = currentFitness[0]
        currentFitness.pop()
        bestFitnessList1.append(bestFitness)  # Modifiable
        currentFitness.append(bestMutationChildFitness)
        index = findWorstinList(population)
        if mutationchild1.fitness < mutationchild2.fitness:
            population[index].bitsString = mutationchild1.bitsString
            population[index].fitness = mutationchild1.fitness
        else:
            population[index].bitsString = mutationchild2.bitsString
            population[index].fitness = mutationchild2.fitness
    return bestFitnessList1


def CCGA4(n, iteration):
    generation = 0
    population = []
    bestFitnessList = []
    for i in range(n):
        subpopulations = []
        for a in range(100):
            b = createIndividual(1)
            subpopulations.append(b)
        population.append(subpopulations)
    fitnessList = []
    for i in range(len(population)):
        for j in range(len(population[i])):
            member = population[i][j]
            integralIndividual = population[i][j].bitsString
            k = 0
            while k != len(population):
                if k == i:
                    k = k + 1
                    continue
                randomIndex = random.randint(0, 99)
                integralIndividual = np.append(integralIndividual, population[k][randomIndex].bitsString)
                k = k + 1
            member.fitness = Ackley(integralIndividual)
            fitnessList.append(member.fitness)
    while generation != iteration:
        generation = generation + 1
        print("current generation: ", generation)
        for i in range(n):
            subpopulation = population[i]
            parent1 = roulette_selection(subpopulation)
            parent2 = roulette_selection(subpopulation)
            # while compare(parent1.bitsString, parent2.bitsString):
            #     parent2 = roulette_selection(subpopulation)
            crossoverchild1, crossoverchild2 = twoPointCrossover(parent1, parent2, int(len(parent1.bitsString) / 16))
            mutationchild1 = Mutation(crossoverchild1, int(len(crossoverchild1.bitsString) / 16))
            mutationchild2 = Mutation(crossoverchild2, int(len(crossoverchild1.bitsString) / 16))
            fitnessList.sort()
            fitnessList.pop()
            if mutationchild1.fitness < mutationchild2.fitness:
                bestMutation = mutationchild1
            else:
                bestMutation = mutationchild2
            integralIndividual = bestMutation.bitsString
            k = 0
            while k != len(population):
                if k == i:
                    k = k + 1
                    continue
                subpopulationTemp = population[k]
                bestIndex = findBestinList(subpopulationTemp)
                integralIndividual = np.append(integralIndividual, subpopulationTemp[bestIndex].bitsString)
                k = k + 1
            bestMutation.fitness = Ackley(integralIndividual)
            fitnessList.append(bestMutation.fitness)
            index = findWorstinList(subpopulation)
            subpopulation[index].bitsString = bestMutation.bitsString
            subpopulation[index].fitness = bestMutation.fitness
        fitnessList.sort()
        bestFitness = fitnessList[0]
        bestFitnessList.append(bestFitness)  # Modifiable
    return bestFitnessList


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def show1(iteration):
    n4 = 30
    List1 = zerolistmaker(iteration)
    # Y1 = normalGA4(n4, iteration)
    # print("Normal GA is done!")
    # Y2 = CCGA4(n4, iteration)
    for i in range(5):
        Y1 = normalGA4(n4, iteration)
        List1 = [i + j for i, j in zip(List1, Y1)]
    for i in range(len(List1)):
        List1[i] = List1[i] / 5
    print("Normal GA is done!")
    List2 = zerolistmaker(iteration)
    for i in range(5):
        Y2 = CCGA4(n4, iteration)
        List2 = [i + j for i, j in zip(List2, Y2)]
    for i in range(len(List2)):
        List2[i] = List2[i] / 5
    X = []
    for i in range(iteration):
        X.append(i)
    test = plt.figure()
    plt.xlabel("Function evaluation")
    plt.ylabel("Best individual")
    plt.title("Ackley Function")
    plt.plot(X, List1, c='k', linestyle='-')
    plt.plot(X, List2, c='k', linestyle='--')
    plt.legend(["Standard GA", "CCGA-1"])
    plt.savefig('Ackley.png')
    plt.show()


# show1(100)
