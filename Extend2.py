import math
import random
import string
import struct

import numpy
import numpy as np
from random import seed
from random import randint
import matplotlib.pyplot as plt
from operator import attrgetter

from Rastrigin import Rastrigin


class Member():

    def __init__(self, bitsString=None):
        self.bitsString = bitsString

        self.sex = 0  # 0 means female and 1 means male
        self.fitness = 0
        self.performanceCount = 0


def createIndividual2(n):
    x = numpy.random.randint(2, size=(16 * n,))
    new_individual = Member(x)
    new_individual.fitness = Rastrigin(new_individual.bitsString)
    return new_individual


def Mutation2(parent, n):
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
    child.fitness = Rastrigin(child.bitsString)
    return child


def twoPointCrossover2(x, y, n):
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
    # print("cross over at: ", p1, p2)
    p = random.uniform(0, 1)
    if p < 0.6:
        # print("crossover happened! ")
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
    newchild1.fitness = Rastrigin(child1)
    newchild2 = Member(child2)
    newchild2.fitness = Rastrigin(child2)
    return newchild1, newchild2


def sexDetermination(population, r):
    n = len(population)
    s = int(n / r)
    for i in range(s):
        for j in range(n):
            member1 = population[j]
            randomnumberList = random.sample(range(100), 4)
            memberList = []
            for k in range(4):
                member = population[randomnumberList[k]]
                memberList.append(member)
            offspringFitnessList = []
            for e in range(4):
                offspring1, offspring2 = twoPointCrossover2(member1, memberList[e], len(member1.bitsString) / 16)
                offspringFitnessList.append(Rastrigin(offspring1.bitsString))
                offspringFitnessList.append(Rastrigin(offspring2.bitsString))
            index3 = offspringFitnessList.index(min(offspringFitnessList))
            if index3 == 0:
                bestOffSpring = offspring1
            else:
                bestOffSpring = offspring2
            if Rastrigin(bestOffSpring.bitsString) < Rastrigin(member.bitsString):
                # population[j].bitsString = bestOffSpring.bitsString
                # population[j].fitness = bestOffSpring.fitness
                population[j].performanceCount = population[j].performanceCount + 1
    totalPerformance = 0
    for i in range(n):
        totalPerformance += population[i].performanceCount
    averagePerformance = totalPerformance / n
    for i in range(n):
        if population[i].performanceCount < averagePerformance:
            population[i].sex = 1
    return population


def CCGAWithSextualSelection(n, iteration, r):
    generation = 0
    population = []
    bestFitnessList = []
    for i in range(n):
        subpopulations = []
        for a in range(100):
            b = createIndividual2(1)
            subpopulations.append(b)
        population.append(subpopulations)
    fitnessList = []
    for i in range(len(population)):
        for j in range(len(population[i])):
            member = population[i][j]
            integralIndividualstring = population[i][j].bitsString
            k = 0
            while k != len(population):
                if k == i:
                    k = k + 1
                    continue
                randomIndex = random.randint(0, 99)
                integralIndividualstring = np.append(integralIndividualstring, population[k][randomIndex].bitsString)
                k = k + 1
            member.fitness = Rastrigin(integralIndividualstring)
            fitnessList.append(member.fitness)
    while generation != iteration:
        generation = generation + 1
        # print("current generation: ", generation)
        for i in range(n):
            subpopulation = population[i]
            if generation == 1:
                subpopulation = sexDetermination(subpopulation, r)
            A = randint(0, 99)
            while subpopulation[A].sex != 0:
                A = randint(0, 99)
            female = subpopulation[A]
            maleList = []
            B = randint(0, 99)
            while subpopulation[B].sex != 1:
                B = randint(0, 99)
            C = randint(0, 99)
            while subpopulation[C].sex != 1:
                C = randint(0, 99)
            D = randint(0, 99)
            while subpopulation[D].sex != 1:
                D = randint(0, 99)
            E = randint(0, 99)
            while subpopulation[B].sex != 1:
                E = randint(0, 99)
            maleList.append(subpopulation[B])
            maleList.append(subpopulation[C])
            maleList.append(subpopulation[D])
            maleList.append(subpopulation[E])
            bestChildList = []
            for e in range(4):
                crossoverchild1, crossoverchild2 = twoPointCrossover2(female, maleList[e],
                                                                      int(len(female.bitsString) / 16))
                mutationchild1 = Mutation2(crossoverchild1, int(len(crossoverchild1.bitsString) / 16))
                mutationchild2 = Mutation2(crossoverchild2, int(len(crossoverchild1.bitsString) / 16))
                if mutationchild2.fitness < mutationchild1.fitness:
                    bestchild = mutationchild2
                    integralIndividualstring2 = mutationchild2.bitsString
                else:
                    bestchild = mutationchild1
                    integralIndividualstring2 = mutationchild2.bitsString
                bestChildList.append(bestchild)
            bestChildList.sort(key=lambda x: x.fitness, reverse=False)
            theChild = bestChildList[0]
            fitnessList.sort()
            fitnessList.pop()
            k = 0
            while k != len(population):
                if k == i:
                    k = k + 1
                    continue
                subpopulationTemp = population[k]
                fitnessTempList = []
                for p in range(len(subpopulationTemp)):
                    fitnessTempList.append(subpopulationTemp[p].fitness)
                bestIndex = fitnessTempList.index(min(fitnessTempList))
                integralIndividualstring2 = np.append(integralIndividualstring2,
                                                      subpopulationTemp[bestIndex].bitsString)
                k = k + 1
            theChild.fitness = Rastrigin(integralIndividualstring2)
            fitnessList.append(theChild.fitness)
            if theChild.fitness <= female.fitness:
                subpopulation[A].bitsString = theChild.bitsString
                subpopulation[A].fitness = theChild.fitness
            else:
                worstIndex = maleList.index(max(maleList, key=attrgetter('fitness')))
                if worstIndex == 0:
                    subpopulation[B].bitsString = theChild.bitsString
                    subpopulation[B].fitness = theChild.fitness
                if worstIndex == 1:
                    subpopulation[C].bitsString = theChild.bitsString
                    subpopulation[C].fitness = theChild.fitness
                if worstIndex == 2:
                    subpopulation[C].bitsString = theChild.bitsString
                    subpopulation[C].fitness = theChild.fitness
                if worstIndex == 3:
                    subpopulation[C].bitsString = theChild.bitsString
                    subpopulation[C].fitness = theChild.fitness
        fitnessList.sort()
        bestFitness = fitnessList[0]
        bestFitnessList.append(bestFitness)
    return bestFitnessList


################################################


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def show2(n, iteration):
    List1 = zerolistmaker(iteration)
    for i in range(5):
        Y1 = CCGAWithSextualSelection(n, iteration, 100)
        List1 = [i + j for i, j in zip(List1, Y1)]
        print("Run has done once: ", i)
    for i in range(len(List1)):
        List1[i] = List1[i] / 5
    print("First half is done!")
    List2 = zerolistmaker(iteration)
    for i in range(5):
        Y2 = CCGAWithSextualSelection(n, iteration, 2)
        List2 = [i + j for i, j in zip(List2, Y2)]
        print("Run has done once: ", i)
    for i in range(len(List2)):
        List2[i] = List2[i] / 5
    X = []
    for i in range(iteration):
        X.append(i)
    plt.xlabel("Function evaluation")
    plt.ylabel("Best individual")
    plt.title("Rastrigin Function")
    plt.plot(X, List1, c='r', linestyle='-')
    plt.plot(X, List2, c='b', linestyle='--')
    plt.legend(["CCGA With Sextual Selection(R = 100)", "CCGA With Sextual Selection(R = 2)"])
    plt.show()


# show2(20, 20000)
