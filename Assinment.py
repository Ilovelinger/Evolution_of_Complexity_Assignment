import random
import string

import numpy as np

randomCharacters = ''.join(random.sample(string.ascii_letters + string.digits + string.punctuation, 1))


# Individual = []
# for index in range(28):
#     Individual.append(''.join(random.sample(string.ascii_letters + string.digits + string.punctuation, 1)))
# print(Individual)


def createIndividual():
    x = []
    for index in range(28):
        x.append(''.join(random.sample(string.ascii_letters + string.digits + string.punctuation, 1)))

    return x


targetlist = "methinks it is like a weasel"
target = []
for index in range(28):
    target.append(targetlist[index])


# def FitnessFuction(x,y):
#     fitness = 0
#     print(target,"sdasdsa")
#     for index in range(28):
#         if x[0, index] == target[0, index]:
#             print('matches',x[0,index], target[0,index])
#             fitness = fitness + 1
#     return fitness

def newFitness(member):
    fitness = 0
    target = "methinks it is like a weasel"
    for i in range(0, len(member)):
        if member[i] == target[i]:
            fitness = fitness + 1
    return fitness


def Mutation(x):
    for i in range(28):
        a = random.uniform(0, 1)
        if (a < (1 / 28)):
            # print('mutate here', i )
            r_char = ''.join(random.sample(string.ascii_letters + string.digits + string.punctuation, 1))
            x[i] = r_char

    return x

ind = createIndividual()
print(ind)
new = Mutation(ind)


# print(newFitness(ind))
print(new)
# print(newFitness(new))





def HillClimber():
    a = createIndividual()
    curr_fitness = newFitness(a)
    while curr_fitness < 28:
        b = Mutation(a)
        if newFitness(b) > newFitness(a):
            print('here')
            a = b.copy()
            curr_fitness = newFitness(b)

HillClimber()

# def genericAlgorithm():
#     for index in range(500):
#         createIndividuals()
