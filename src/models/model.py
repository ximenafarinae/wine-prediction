from src.utils.prediction.utils import *

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        # Construir una red neuronal con pesos y sesgos iniciados aleatoriamente
        self.w_hidden = np.random.rand(hidden_layer_size, input_size)
        self.w_output = np.random.rand(output_size, hidden_layer_size)

        self.b_hidden = np.random.rand(hidden_layer_size, output_size)
        self.b_output = np.random.rand(1, output_size)

    # Derivadas de las funciones de activación

    def forward_prop(self, X):
        Z1 = self.w_hidden @ X + self.b_hidden
        A1 = relu(Z1)

        Z2 = self.w_output @ A1 + self.b_output
        A2 = logistic(Z2)

        return Z1, A1, Z2, A2

    def backward_prop(self, x, y, z1, a1, z2, a2, learning_rate):
        # Derivada del costo con respecto a la salida A2
        dC_dA2 = 2 * a2 - 2 * y
        # Derivada de la salida A2 con respecto a Z2 (siendo Z2 el calculo de salida antes de pasar por la funcion de activacion)
        dA2_dZ2 = d_logistic(z2)
        # Derivada de Z2 con respecto a la salida A1 (capa oculta)
        dZ2_dA1 = self.w_output
        # Derivada de Z2 con respecto al peso de la capa de salida.
        dZ2_dW2 = a1
        # Derivada de Z2 con respecto al sesgo en la capa de salida.
        dZ2_dB2 = 1
        # Derivada de A1 (salida de la capa oculta) con respecto a Z1 (siendo Z1 la salida de la capa oculta antes de pasar por la funcion de activacion)
        dA1_dZ1 = d_relu(z1)
        # Derivada de Z1 con respecto al peso de la capa oculta.
        dZ1_dW1 = x
        # Derivada de Z1 con respecto al sesgo de la capa oculta.
        dZ1_dB1 = 1

        dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T

        dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2

        dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1

        dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T

        dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

        self.w_output -= learning_rate * dC_dW2
        self.b_output -= learning_rate * dC_dB2
        self.w_hidden -= learning_rate * dC_dW1
        self.b_hidden -= learning_rate * dC_dB1