from functions import sigmoid
import random
class Neuron(object):

    def __init__(self, inputs=None, weights=None, output_function=sigmoid, input_connected=None):
        self.inputs = [] if inputs is None else inputs
        self.weights = [] if weights is None else weights
        self.output_function = output_function
        self.input_connected = [] if input_connected is None else input_connected
    
    def input(self, input, weight=0):
        self.inputs.append(input)
        if weight:
            self.weights.append(weight)
    
    def input_from_connected(self):
        self.inputs = []
        for neuron in self.input_connected:
            self.inputs.append(neuron.output())

    def _sum(self):
        sum = 0
        for i in range(len(self.inputs)):
            sum += self.inputs[i] * self.weights[i]
        return sum

    def output(self):
        return self.output_function(self._sum())
        

class NeuralNetwork(object):

    def __init__(self):
        self.input_neurons = []
        self.output_neurons = []
        self.score = 0

    def add_input_neuron(self, weights, quantity=1, output_function=sigmoid):
        for i in range(quantity):
            self.input_neurons.append(Neuron(weights=weights, output_function=output_function))
    
    def add_input(self, inputs, index=-1):
        if index == -1:
            for i in range(len(self.input_neurons)):
                if not self.input_neurons[i].inputs:
                    index = i
                    break
            if index == -1:
                return
        self.input_neurons[index].inputs = inputs
    
    def clear_inputs(self):
        for neuron in self.input_neurons:
            neuron.inputs = []

    def add_output_neuron(self, weights, input_connected=None, output_function=sigmoid):
        input_connected = [] if input_connected is None else input_connected
        input_connected = [self.input_neurons[i] for i in input_connected]
        neuron = Neuron(weights=weights, input_connected=input_connected, output_function=output_function)
        self.output_neurons.append(neuron)
    
    def output(self):
        outputs = []
        for neuron in self.output_neurons:
            neuron.input_from_connected()
            outputs.append(neuron.output())
        return outputs
    
    def _get_new_input_weights(self, weights1, weights2):
        learn_factor = random.uniform(-0.02, 0.02)
        new_input_weights= []
        for i in range(len(weights1)):
            choice = random.randrange(1, 3)
            if choice == 1:
                new_input_weights.append(weights1[i] + learn_factor)
            else:
                new_input_weights.append(weights2[i] + learn_factor)
        return new_input_weights


    def cross(self, other):
        input_neurons1 = self.input_neurons
        input_neurons2 = other.input_neurons
        output_neurons1 = self.output_neurons
        output_neurons2 = other.output_neurons
        new_networks = []
        for i in range(4):
            nn = NeuralNetwork()
            for j in range(len(input_neurons1)):
                new_weights = self._get_new_input_weights(input_neurons1[j].weights, input_neurons2[j].weights)
                nn.add_input_neuron(new_weights)

            for j in range(len(output_neurons1)):
                connected_indexes = [input_neurons1.index(x) for x in output_neurons1[j].input_connected]
                new_weights = self._get_new_input_weights(output_neurons1[j].weights, output_neurons2[j].weights)

                nn.add_output_neuron(new_weights, connected_indexes)
            new_networks.append(nn)
        return new_networks