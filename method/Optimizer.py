import math
import numpy as np


class Optimizer:
    """
    Simulation training by using a variety of optimizers
    """

    def __init__(self, objective, derivative, steps: int, start_point: list):
        """
        :param objective: Objective function
        :param derivative: Derivative of the objective function
        :param steps: Number of optimiser executions
        :param start_point: Starting point for gradient descent
        """
        self.objective = objective
        self.derivative = derivative
        self.steps = steps
        self.start_point = start_point

    def sgd(self, alpha: float, gamma: float):
        """
        Optimize with SGD, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'SGD step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # Update
                x[i] = x[i] - alpha * g[i]

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'SGD step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list

    def momentum(self, alpha, gamma, beta_1):
        """
        Optimize with Momentum, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :param beta_1: First-order moment estimation parameters
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'Momentum step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')
        # Initialize first moments
        m = [0.0 for _ in range(len(self.start_point))]

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # First moment update
                m[i] = beta_1 * m[i] + (1.0 - beta_1) * g[i]
                # Bias correct
                m_hat = m[i] / (1.0 - beta_1 ** (t + 1))
                # Update
                x[i] = x[i] - alpha * m_hat

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'Momentum step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list

    def rmsprop(self, alpha: float, gamma: float, beta_2: float, epsilon: float = 1e-8):
        """
        Optimize with RMSprop, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :param beta_2: Second-order moment estimation parameters
        :param epsilon: Prevent the denominator from being zero
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'RMSprop step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')
        # Initialize first and second moments
        v = [0.0 for _ in range(len(self.start_point))]

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # Second moment update
                v[i] = beta_2 * v[i] + (1.0 - beta_2) * g[i] ** 2
                # Bias correct
                v_hat = v[i] / (1.0 - beta_2 ** (t + 1))
                # Update
                x[i] = x[i] - alpha * g[i] / (math.sqrt(v_hat) + epsilon)

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'RMSprop step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list

    def adam(self, alpha: float, gamma: float, beta_1: float, beta_2: float, epsilon: float = 1e-8):
        """
        Optimize with Adam, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :param beta_1: First-order moment estimation parameters
        :param beta_2: Second-order moment estimation parameters
        :param epsilon: Prevent the denominator from being zero
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'Adam step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')
        # Initialize first and second moments
        m = [0.0 for _ in range(len(self.start_point))]
        v = [0.0 for _ in range(len(self.start_point))]

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # First and second moment update
                m[i] = beta_1 * m[i] + (1.0 - beta_1) * g[i]
                v[i] = beta_2 * v[i] + (1.0 - beta_2) * g[i] ** 2
                # Bias correct
                m_hat = m[i] / (1.0 - beta_1 ** (t + 1))
                v_hat = v[i] / (1.0 - beta_2 ** (t + 1))
                # Update
                x[i] = x[i] - alpha * m_hat / (math.sqrt(v_hat) + epsilon)

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'Adam step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list

    def signum(self, alpha, gamma, beta_1):
        """
        Optimize with Signum, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :param beta_1: First-order moment estimation parameters
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'Signum step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')
        # Initialize first moments
        m = [0.0 for _ in range(len(self.start_point))]

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # First moment update
                m[i] = beta_1 * m[i] + (1.0 - beta_1) * g[i]
                # Bias correct
                m_hat = m[i] / (1.0 - beta_1 ** (t + 1))
                # Update
                x[i] = x[i] - alpha * np.sign(m_hat)

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'Signum step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list

    def lion(self, alpha, gamma, beta_1, beta_3):
        """
        Optimize with Lion, returns a list of all training tracks and the corresponding loss
        :param alpha: Learning rate
        :param gamma: Decline parameter of learning rate, the smaller the learning rate decline faster
        :param beta_1: First-order moment estimation parameters
        :param beta_3: Weighting factors for updating
        :return: Points passed during the optimisation process and their corresponding loss values
        """
        # Initialise the list of return values
        track_list = []
        loss_val_list = []
        # Define the starting point
        x = self.start_point.copy()
        # Add the coordinates of the initial point and the loss value to the return list
        track_list.append(x.copy())
        loss_val_list.append(self.objective(x))
        print('Training')
        print(f'Lion step: 0  point:{track_list[0]}  loss:{loss_val_list[0]}')
        # Initialize first moments
        m = [0.0 for _ in range(len(self.start_point))]

        # Run the gradient descent updates
        for t in range(self.steps):
            # calculate gradient g(t)
            g = self.derivative(x)

            # Update parameters one by one
            for i in range(len(self.start_point)):
                # Bias correct
                m_hat = m[i] / (1.0 - beta_1 ** (t + 1))
                # Update
                x[i] = x[i] - alpha * np.sign(beta_3 * m_hat + (1.0 - beta_3) * g[i])
                # First moment update
                m[i] = beta_1 * m[i] + (1.0 - beta_1) * g[i]

            # Keep track of solutions
            track_list.append(x.copy())
            loss_val_list.append(self.objective(x))
            print(f'Lion step: {t + 1}  point:{track_list[t + 1]}  loss:{loss_val_list[t + 1]}')
            # Simulated annealing algorithm
            alpha = alpha * gamma ** t

        return track_list, loss_val_list
