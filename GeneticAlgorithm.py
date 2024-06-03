import numpy as np
from typing import Callable

def activationFunctions(name):
    def tanh(x):
        return np.tanh(x)
    
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def relu(x):
        return x * (x > 0)
    
    if(name == "Tanh"):
        return tanh
    elif(name == "Sigmoid"):
        return sigmoid
    else:
        return relu
    
class Genome:
    def __init__(self, genome: np.ndarray):
        self.fitness = 0
        self.genome = np.array(genome)

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        return self.fitness >= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def __repr__(self):
      return f"Genome({self.genome})"
    
    def __str__(self):
      return f"Genome: {self.genome} Fitness: {self.fitness}"

class neuralNetwork:
    def __init__(self, networkArchitecture: np.ndarray,
                outputActivationFunction: str = "Tanh",):
        
        self.architecture = np.array(networkArchitecture)
        self.ActivationFunction = activationFunctions("Relu")
        self.outputActivationFunction = activationFunctions(outputActivationFunction)

        self.totalBiases = np.sum(self.architecture[1:])
        self.totalWeights = np.sum(np.multiply(self.architecture[:-1], self.architecture[1:]))
        


    def feedForward(self, genome: Genome, inputs: np.ndarray) -> np.ndarray:
        values = np.array(inputs)
        assert values.size == self.architecture[0], f"There should be as many inputs as there are input neurons in the neural network ({self.architecture[0]})"
        for i, _ in enumerate(self.architecture[1:], 1):
            layerBiases = genome.genome[np.sum(self.architecture[1:i], dtype=int) : np.sum(self.architecture[1:i+1], dtype=int)]
            layerWeights = genome.genome[self.totalBiases + np.sum(np.multiply(self.architecture[:i-1], self.architecture[1:i]), dtype=int) : self.totalBiases + np.sum(np.multiply(self.architecture[:i], self.architecture[1:i+1]), dtype=int)].reshape(self.architecture[i], self.architecture[i-1])
            values = layerBiases + np.sum(values*layerWeights,axis=1)
            # print(f"layerBiases: {layerBiases}")
            # print(f"layerWeights: {layerWeights}")
            # print(f"WeightsPart: {np.sum(values*layerWeights,axis=1)}")
            # print(values)
            
            if i == self.architecture.size -1:
                values = np.frompyfunc(lambda x: self.outputActivationFunction(x),1,1)(values)
            else:
                values = np.frompyfunc(lambda x: self.ActivationFunction(x),1,1)(values)
        return values
            

            
        

class GeneticAlgorithm:
    def __init__(self, 
                fitnessFunction: Callable[[any,any],None],
                fitnessThreshold: np.double,
                neuralNetwork: neuralNetwork,
                populationSize: int = 50,
                maxGenerations: int = 50,
                elitism: int = 5,
                weightRange: np.ndarray = np.array([-30, 30]),
                biasRange: np.ndarray = np.array([-30, 30]),
                weightMutateRate: float = 0.05,
                biasMutateRate: float = 0.05,
                weightMutateRange: np.ndarray = np.array([-3, 3]),
                biasMutateRange: np.ndarray = np.array([-3, 3])
                ):
        
        self.fitnessFunction = fitnessFunction
        self.fitnessThreshold = fitnessThreshold
        self.MaxFitness = 0

        self.nn = neuralNetwork

        self.populationSize = populationSize
        self.generation = 1
        self.maxGeneration = maxGenerations
        self.elitism = elitism

        self.weightRange = np.array(weightRange)
        self.biasRange = np.array(biasRange)
        self.weightMutateRate = weightMutateRate
        self.biasMutateRate = biasMutateRate
        self.weightMutateRange = np.array(weightMutateRange)
        self.biasMutateRange = np.array(biasMutateRange)

        self.randomGenerator = np.random.default_rng()
        
        self.runAlgorithm()

    def runAlgorithm(self):
        self.genomes = self.generateGenomes()
        self.fitnessFunction(self.genomes, self.nn)
        self.MaxFitness = np.max(self.genomes).fitness
        while(self.MaxFitness < self.fitnessThreshold and self.generation <= self.maxGeneration):
            self.generation += 1
            self.crossover()
            self.mutate()
            self.fitnessFunction(self.genomes, self.nn)
            self.MaxFitness = np.max(self.genomes).fitness
            print(self.generation)
        print(f"{np.max(self.genomes)} Generation: {self.generation}")


    def generateGenomes(self) -> np.ndarray:
        genomes = []
        for _ in range(self.populationSize):
            biases = self.randomGenerator.uniform(low=self.biasRange[0], high=self.biasRange[1], size = self.nn.totalBiases)
            weights = self.randomGenerator.uniform(low=self.weightRange[0], high=self.weightRange[1], size = self.nn.totalWeights)
            genome = np.append(biases, weights)
            genomes.append(Genome(genome))
            
        return np.array(genomes)
    

    def crossover(self):
        sortedGenomes = np.sort(self.genomes)
        genome_fitnesses = np.frompyfunc(lambda x: x.fitness,1,1)(sortedGenomes)
        cumulative_fitness = np.cumsum(genome_fitnesses)

        if self.elitism > 0:
            newGenomes = sortedGenomes[-self.elitism:]
        else:
            newGenomes = np.array([], dtype = sortedGenomes.dtype)

        while(newGenomes.size < self.populationSize):
            parentGenomes = self.selection(sortedGenomes, cumulative_fitness)
            mask = self.randomGenerator.integers(2, size = parentGenomes[0].genome.size)
            genome1 = np.where(mask, parentGenomes[0].genome, parentGenomes[1].genome)
            genome2 = np.where(mask, parentGenomes[1].genome, parentGenomes[0].genome)
            newGenomes = np.append(newGenomes, [Genome(genome1), Genome(genome2)])

        self.genomes = newGenomes[:self.populationSize]


    def selection(self, sortedGenomes: np.ndarray, cumulative_fitness: np.ndarray):
        parentIdx = [0, 0]
        while(parentIdx[0] == parentIdx[1]):
            selecter = self.randomGenerator.uniform(low = 0, high = cumulative_fitness[-1], size = 2)
            parentIdx = np.searchsorted(cumulative_fitness, selecter, side='right')
        return sortedGenomes[parentIdx]


    def mutate(self):
        for genome in self.genomes:
            biasMutations = self.randomGenerator.uniform(low=self.biasMutateRange[0], high=self.biasMutateRange[1], size = self.nn.totalBiases)
            mutateBias = self.randomGenerator.uniform(low=0, high=1, size = self.nn.totalBiases)
            biasMutations = np.where(mutateBias <= self.biasMutateRate, biasMutations, 0)

            weightMutations = self.randomGenerator.uniform(low=self.weightMutateRange[0], high=self.weightMutateRange[1], size = self.nn.totalWeights)
            mutateWeight = self.randomGenerator.uniform(low=0, high=1, size = self.nn.totalWeights)
            weightMutations = np.where(mutateWeight <= self.weightMutateRate, weightMutations, 0)

            newBiases = np.clip(genome.genome[:self.nn.totalBiases] + biasMutations, self.biasRange[0], self.biasRange[1])
            newWeights = np.clip(genome.genome[self.nn.totalBiases:] + weightMutations, self.weightRange[0], self.weightRange[1])

            genome.genome = np.append(newBiases,newWeights)



# gene = Genome([10,20,30,1,2,3,4,5,6,7,8])
# nn1 = neuralNetwork([3,2,1])

# print(nn1.feedForward(gene, [1,2,1]))

# def generateGenomes(self):
#         genomes = []
#         for _ in range(self.populationSize):
#             genome = []
#             for prev_layer, layer in zip(self.nn.architecture[:-1], self.nn.architecture[1:]):
#                 # Generate biases for each neuron in the layer
#                 biases = self.randomGenerator.uniform(low=self.nn.biasRange[0], high=self.nn.biasRange[1], size=layer)
#                 # Generate weights for each connection from the previous layer to the current layer
#                 weights = self.randomGenerator.uniform(low=self.nn.weightRange[0], high=self.nn.weightRange[1], size=(layer, prev_layer))
#                 # Interleave biases and weights for each neuron
#                 interleaved = np.column_stack((biases, weights))
#                 genome.extend(interleaved.flatten())
#             genomes.append(Genome(genome))
            
#         return genomes