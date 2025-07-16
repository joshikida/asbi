import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import matplotlib.pyplot as plt

from components import Oracle
import pts
import data_loader


class ActiveLearner:
    """Run AL loop"""
    def __init__ (self,x_space,acqui_policy,simulation_params=1):
        if simulation_params != 1 and simulation_params != 5: 
            raise RuntimeError("Valid simulation parameter choices are 1 and 5")
            
        self.acquisition_policy=acqui_policy
        self.simulation_params =simulation_params
        self.true_params, self.true_images = data_loader.load(simulation_params)
        
        self.oracle = Oracle(self.true_images,self.true_params) ## TODO NOW TRACE
        self.x_space = x_space
        self.X_train = []
        self.y_train = []
        # for visualization
        self.y_true = [self.oracle.evaluate(x,batch_size=5) for x in self.x_space]

    def _plot_iteration(self, iteration, y_pred, sigma, next_design_x):
        """
        Helper function to plot the state of the learning process
        """
        plt.clf()

        # Plot the true performance (now MSE)
        plt.plot(self.x_space, self.y_true, 'r--', label='True Performance (MSE)')

        # Plot the surrogate's prediction and confidence interval
        plt.plot(self.x_space, y_pred, 'b-', label='Surrogate Model Prediction')
        plt.fill_between(self.x_space.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                         alpha=0.2, color='blue', label='95% Confidence Interval')

        # Plot the points already queried
        plt.scatter(self.X_train, self.y_train, c='black', s=50, zorder=10,
                    edgecolor='white', label='Queried Points')

        # Highlight the next point to be queried
        plt.scatter(next_design_x, 0, c='gold', s=150, zorder=11,
                    edgecolor='black', marker='*', label='Next Point to Query')

        plt.title(f"Active Learning Progress: Iteration {iteration}")
        plt.xlabel("Instrument Design Parameter (Noise Std. Dev.)")
        # --- THIS LABEL HAS BEEN UPDATED ---
        plt.ylabel("Performance Metric (Mean Squared Error)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':')
        plt.ylim(bottom=0)
        plt.pause(0.5)




    def run (self,initial_points=3,total_queries=15,plot_progress=False):
        if plot_progress:
            plt.figure(figsize=(12,7))
        print("\n == START ACTIVE LEARNING ==\n")
        print(f"Selected acquisition policy {self.acquisition_policy.__name__}")
        print(f"Performing {initial_points} initial random queries")
        
        # cold start, simulate on some random points in design space
        initial_indices = np.random.choice(len(self.x_space),initial_points,replace=False)
        for i in initial_indices:
            design_x = self.x_space[i]
            performance_y = self.oracle.evaluate(design_x)
            self.X_train.append(design_x)
            self.y_train.append(performance_y)

        # main loop
        for i in range(initial_points,total_queries):
            print(f"Iteration {i+1}/{total_queries}")

            ## train surrogate [GP] on data that has been labeled thus far
            """
            kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
            gp_surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10 # Increasing restarts can also help find a better fit
            )
            gp_surrogate.fit(np.array(self.X_train), np.array(self.y_train))
            """


            """
            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
            gp_surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            gp_surrogate.fit(np.array(self.X_train), np.array(self.y_train))
            """

            kernel = ConstantKernel(1.0, (1e-4, 1e2)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0)) \
                     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-1))

            gp_surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=20 
            )
            gp_surrogate.fit(np.array(self.X_train), np.array(self.y_train))

            ## select next design with given acquisition function
            y_pred, sigma = gp_surrogate.predict(self.x_space, return_std=True)
            next_design_x = self.acquisition_policy(y_pred,sigma,self.x_space,self.y_train)

            if plot_progress:
                self._plot_iteration(i+1,y_pred,sigma,next_design_x)

            print(f"  Surrogate trained on {len(self.X_train)} points.")
            print(f"  Acquisition policy selected next design: noise_std = {next_design_x}")

            ## oracle on selected design
            new_y = self.oracle.evaluate(next_design_x)
            print(f"  Oracle returned performance (uncertainty): {new_y:.4f}")

            ## add to training data
            self.X_train.append(next_design_x)
            self.y_train.append(new_y)

        if plot_progress: plt.show()
        return np.array(self.X_train), np.array(self.y_train)
    
