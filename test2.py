from learning import NeuralNetwork
import random
import math

def test(nets, print_=False):
    for net in nets:
        score = 0
        # 1 and 1 = 1
        net.add_input([1])
        net.add_input([1])
        value_true1 = net.output()[0]
        score -= 1 - value_true1
        net.clear_inputs()

        net.add_input([1])
        net.add_input([0])
        value_true2 = net.output()[0]
        score -= 1 - value_true2
        net.clear_inputs()

        net.add_input([0])
        net.add_input([1])
        value_true3 = net.output()[0]
        score -= 1 - value_true3
        net.clear_inputs()

        net.add_input([0])
        net.add_input([0])
        value_false = net.output()[0]
        score -= (0 + value_true1)
        net.clear_inputs()

        score -= abs(value_true1 - value_true2 - value_true3)

        score += value_true1 - value_false
        score += value_true2 - value_false
        score += value_true3 - value_false

        net.score = score

        if print_:
            print "1 and 1", value_true1
            print "1 and 0", value_true2
            print "0 and 1", value_true3
            print "0 and 0", value_false

def same(x):
    return x

if __name__ == "__main__":
    # generating
    nns = []
    for i in range(1000):
        nn = NeuralNetwork()
        
        nn.add_input_neuron([1], output_function=same)
        nn.add_input_neuron([1], output_function=same)

        w = []
        w2 = []

        for i in range(16):
            w.append([random.uniform(-1, 1), random.uniform(-1, 1)])
            w2.append(random.uniform(-1, 1))
        nn.add_hidden_layer(w)

        for i in range(8):
            w.append([random.uniform(-1, 1), random.uniform(-1, 1)])
            w2.append(random.uniform(-1, 1))
        nn.add_hidden_layer(w)

        nn.add_output_neuron(w2, output_function=same)
        nns.append(nn)

        

    for i in range(150):
        test(nns)
        nns.sort(key=lambda x: x.score)
        # x =  [x.input_neurons for x in nns]
        # print [x[i][j].weights for j in range(len(x[i])) for i in range(len(x))]
        print nns[-2].score
        # test([nns[0]], True)
        nns = nns[750:1000]
        new_nns = []
        for nn in nns:
            i = random.randint(0, 249)
            new_nns += nn.cross(nns[i])
        nns = [x for x in new_nns]

    test([nns[-2]], True)