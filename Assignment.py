import random
import string

import numpy
import numpy as np
from random import seed
from random import randint

chars = []
for i in range(32, 127):
    chars.append(chr(i))


def createIndividual():
    x = []
    for index in range(28):
        x.append(''.join(random.sample(string.ascii_letters + string.digits + string.punctuation, 1)))

    return x


targetlist = "methinks it is like a weasel"
target = []
for index in range(28):
    target.append(targetlist[index])


def fitnessFunction(member):
    fitness = 0
    target = "methinks it is like a weasel"
    for i in range(0, len(member)):
        if member[i] == target[i]:
            fitness = fitness + 1
    return fitness


def Mutation(parent: str):
    parent = list(parent)
    child = [0] * 28

    for i in range(0, 28):
        p = random.uniform(0, 1)
        if (p) < (1 / 28):
            child[i] = random.choice(chars)
        else:
            child[i] = parent[i]

    return child


def HillClimber():
    a = createIndividual()
    curr_fitness = fitnessFunction(a)
    while curr_fitness < 28:
        b = Mutation(a)
        # print(curr_fitness)
        if fitnessFunction(b) > fitnessFunction(a):
            a = b
            curr_fitness = fitnessFunction(a)
            print(a)


# HillClimber()


def genericAlgorithm():
    population = []
    for a in range(500):
        b = createIndividual()
        population.append(b)

    curr_fitness = fitnessFunction(population[randint(0, 499)])
    generation = 0
    while curr_fitness < 28:
        A = randint(0, 499)
        B = randint(0, 499)
        if fitnessFunction(population[A]) > fitnessFunction(population[B]):
            parent1 = population[A]
        else:
            parent1 = population[B]

        child = Mutation(parent1)
        curr_fitness = fitnessFunction(child)
        print(curr_fitness)
        A = randint(0, 499)
        B = randint(0, 499)
        if fitnessFunction(population[A]) > fitnessFunction(population[B]):
            population[B] = child
            generation = generation + 1
        else:
            population[A] = child
            generation = generation + 1
        print(generation)


# genericAlgorithm()


def Crossover(x, y):
    parent1 = x
    parent2 = y
    child = [0] * 28
    for i in range(28):
        p = random.uniform(0, 1)
        if p < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return child


def genericAlgorithmWithCrossover():
    population = []
    for a in range(500):
        b = createIndividual()
        population.append(b)

    curr_fitness = fitnessFunction(population[randint(0, 499)])
    generation = 0
    while curr_fitness < 28:
        A = randint(0, 499)
        B = randint(0, 499)
        if fitnessFunction(population[A]) > fitnessFunction(population[B]):
            parent1 = population[A]
        else:
            parent1 = population[B]

        A = randint(0, 499)
        B = randint(0, 499)
        if fitnessFunction(population[A]) > fitnessFunction(population[B]):
            parent2 = population[A]
        else:
            parent2 = population[B]

        crossoverchild = Crossover(parent1, parent2)
        mutationchild = Mutation(crossoverchild)
        curr_fitness = fitnessFunction(mutationchild)
        print(curr_fitness)
        A = randint(0, 499)
        B = randint(0, 499)
        if fitnessFunction(population[A]) > fitnessFunction(population[B]):
            population[B] = mutationchild
            generation = generation + 1
        else:
            population[A] = mutationchild
            generation = generation + 1
        print(generation)


genericAlgorithmWithCrossover()
