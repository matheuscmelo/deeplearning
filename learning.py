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
        if not self.inputs:
            self.input_from_connected()
        return self.output_function(self._sum())
        

class NeuralNetwork(object):

    def __init__(self):
        self.input_neurons = []
        self.output_neurons = []
        self.hidden_layers = []
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
    
    def add_hidden_layer(self, weights, output_function=sigmoid):
        layer = []
        connected_to = self.input_neurons if not self.hidden_layers else self.hidden_layers[-1]
        for weight in weights:
            layer.append(Neuron(weights=weight, input_connected=self.input_neurons, output_function=output_function))
        self.hidden_layers.append(layer)

    def clear_inputs(self):
        for neuron in self.input_neurons:
            neuron.inputs = []
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.inputs = []
        for neuron in self.output_neurons:
            neuron.inputs = []

    def add_output_neuron(self, weights, input_connected=None, output_function=sigmoid):
        input_connected = self.hidden_layers[-1]
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
        hidden_layers1 = self.hidden_layers
        hidden_layers2 = other.hidden_layers
        new_networks = []
        for i in range(4):
            nn = NeuralNetwork()
            for j in range(len(input_neurons1)):
                new_weights = self._get_new_input_weights(input_neurons1[j].weights, input_neurons2[j].weights)
                nn.add_input_neuron(new_weights, output_function=input_neurons1[j].output_function)

            for j in range(len(hidden_layers1)):
                new_weights = []
                for k in range(len(hidden_layers1[j])):
                    new_weights.append(self._get_new_input_weights(hidden_layers1[j][k].weights, hidden_layers2[j][k].weights))
                nn.add_hidden_layer(new_weights, output_function=hidden_layers1[j][k].output_function)

            for j in range(len(output_neurons1)):
                new_weights = self._get_new_input_weights(output_neurons1[j].weights, output_neurons2[j].weights)
                nn.add_output_neuron(new_weights, output_function=output_neurons1[j].output_function)

            new_networks.append(nn)
        return new_networks