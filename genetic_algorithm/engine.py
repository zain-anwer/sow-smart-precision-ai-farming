import random
import numpy as np
from deap import base, creator, tools, algorithms
from data.crops import NUM_CROPS
from vision.zone_detection import detect_zones
from genetic_algorithm.fitness import get_fitness_func


def run_ga(soil_quality,moisture_level,ndvi,cx_prob,
           mut_prob,n=10,n_gen=100,pop_size=300,progress_callback=None):
    

    zone_labels = detect_zones(soil_quality,moisture_level,ndvi)
    print(zone_labels)

    # chromosome is going to be an nxn flat array initialized with crop indexes

    CHROM_LEN = n * n

    # hasattr creates classes at runtime if they don't already exist
    # create function takes the name, parent class, and some other customizable attribute of that class

    # weights are a tuple because of multi-objective ga
    # the first value would be maximized (1) and the second one would be minimized (-1)
    if not hasattr(creator,'FitnessMulti'):
        creator.create('FitnessMulti',base.Fitness,weights=(1,-1))

    if not hasattr(creator,'Individual'):
        # fitness will contain an object named values which would be a tuple
        creator.create('Individual',list, fitness = creator.FitnessMulti)

    # base.ToolBox() returns an object that allows us to register callable operations

    toolbox = base.Toolbox()

    # creating the chromosome generation function
    # toolbox.register(alias,func,*args) --> makes the alias the new function wrapper
    # toolbox.initRepeat(container,func,n) --> fills the container by running func n times

    # guessing a random crop index
    toolbox.register('get_crop_idx',random.randint,0,NUM_CROPS - 1)
    
    # generating a matrix of crop indexes
    toolbox.register('generate_chromosome',tools.initRepeat,creator.Individual,toolbox.get_crop_idx,n=100)

    # generating a population
    # leaving out the third argument for initRepeat which will be provided at runtime (population size)
    toolbox.register('generate_population',tools.initRepeat,list,toolbox.generate_chromosome)

    evaluate = get_fitness_func(soil_quality,moisture_level,ndvi,zone_labels)
    toolbox.register('evaluate',evaluate)

    # takes two individuals at runtime and returns a tuple of modified ones
    toolbox.register('mate',tools.cxTwoPoint)

    # takes an individual at runtime and modifies it based on the probability
    toolbox.register('mutate',tools.mutUniformInt, low = 0, up = NUM_CROPS - 1, indpb = 0.05)

    # takes population and k (new population size) at runtime
    toolbox.register('selection',tools.selNSGA2)

    # generating the population:
    population = toolbox.generate_population(pop_size)

    # to keep track of the single best solution based on fitness
    # its kinda an array which updates based on fitness
    hof = tools.HallOfFame(1)
    pareto_front = tools.ParetoFront()

    # log to append values of best and average fit per generation
    log = []

    for gen in range(n_gen):
        
        # pipeline for running population -> crossover -> mutation
        offspring = algorithms.varAnd(population,toolbox,cx_prob,mut_prob)

        fitness_values = toolbox.map(evaluate,offspring)
        
        # give the fitness values to the individual/chromosome objects
        for fitness_value, ind in zip(fitness_values,offspring):
            ind.fitness.values = fitness_value
        
        population = toolbox.selection(offspring,k=len(population))

        hof.update(population)
        pareto_front.update(population)

        best_fit = hof[0].fitness.values[0]
        average_fit = np.mean([idn.fitness.values[0] for idn in population])

        log.append({'gen':gen+1,'best_fit':best_fit,'average_fit':average_fit})

        # function to send back data to front end for continuous display
        if progress_callback:
            progress_callback(gen,n_gen,best_fit,hof[0])

    return np.array(hof[0]).reshape(n,n), log, pareto_front