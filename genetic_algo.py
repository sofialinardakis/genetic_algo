from functools import partial
import time
from typing import List, Callable, Tuple
from random import choices, randint, randrange, random
from collections import namedtuple

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

#ITEMS
Thing = namedtuple("Thing", ["name", "value", "weight"])

things = [
    Thing("Laptop", 500, 2200),
    Thing("Headphones", 150, 160),
    Thing("Coffee Mug", 60, 350),
    Thing("Notepad", 40, 333),
    Thing("Water Bottle", 30, 192)
]

more_things = [
    Thing("Mints", 5, 25),
    Thing("Socks", 10, 38),
    Thing("Tissues", 15, 80),
    Thing("Phone", 500, 200),
    Thing("Baseball Cap", 100, 70)
] + things


#Genome: list of 0/1 (don't pick/pick)
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

#Population: bunch of genomes (random guesses/solutions)
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

#Fitness function: calculates how 'good a genome is
"""
Loops over items: 
    if bit = 1 -> add weight + value #we take and add to backpack
    if weight goes over limit -> fitness = 0 #not valid
otherwise, returns total value:
    higher fitness = better backpack
"""
def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same length")
    
    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0
            
    return value


#Selection: chooses 2 genomes to be parents
"""
    Better fitness = higher chance of being picked
        Survival of the fittest

* k=2; randomly picks 2 genomes from population
        not all genomes have same chance:
        · weights added so stronger genomes (+fitness) = more chance
"""
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

#Crossover / Breeding
"""
Cut both genomes at a random point, swap halves
Produces 2 children (mixes parent DNA)
"""
def single_point_crossover(a: Genome, b:Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be same length")
    
    length = len(a)
    if length < 2:
        return a, b
    
    p = randint(1, length-1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

#Mutation
"""
Flips a random bit(0 <-> 1) with chance
Keep diverse population & prevents stuck algo
* randrange picks a random index in genome, this is flipped if random() <= prob
"""
def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    
    return genome


##MAIN LOOP (evolution process)
"""
Sorts by fitness (best to worst)
If best fitness >= goal -> stop
otherwise:
    keep 2 best genomes
    fill rest by breeding + mutating parents
repeat until max generations or solutions found
    population 'evolves' each loop
"""
def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse = True
        )

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        # new gen: size = ((past gen size / 2) -1) + 2 elite genomes
        for j in range(int(len(population)/2)-1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b] 

        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )

    return population, i

##RUNNING THE ALGORITHM
"""
create initial population
use fitness with item list + weight limit
stop if fitness target hit or generation limit
*partial() returns a new callable with set arguments
"""
start = time.time()
population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(more_things)
    ), 
    fitness_func=partial(
        fitness, things=more_things, weight_limit=3000
    ), 
    fitness_limit=1310,
    generation_limit = 100
)
end = time.time()

#Converts genomes back to items, shows chosen items with best genome
def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]

    return result

#Print results
print(f"number of generations: {generations}")
print(f"time: {end - start}s")
print(f"best solution: {genome_to_things(population[0], more_things)}")


"""
1) Start with a random population of possible backpacks (genomes).
2) Evaluate fitness of each one (total value, if weight ≤ limit)
3) Select parents based on fitness (better ones more likely to reproduce).
4) Crossover parents to make children (mixing solutions).
5) Mutate some children (small random tweaks).
6) Repeat this process for many generations.
7) Eventually → best genome found = best set of items under weight limit.
"""