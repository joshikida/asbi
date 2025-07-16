import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

class NaiveNoiseInf:
    """Fake inference model for 1param noise design"""
    def __init__ (self,simulation_params):
        self.simulation_params = simulation_params
        print(f"Initialized naive noise inference pipeline for {simulation_params} parameters")

    def infer_posterior (self,noisy_img:np.ndarray, true_img:np.ndarray, simulation_params_vector) -> norm:
        """Take noisy img and corresponding ground truth to 'infer' posterior"""
        if self.simulation_params == 1:
            # posterior uncertainty increases with image noise
            image_noise_level = np.std(noisy_img - true_img)
            posterior_std = self.base_uncertainty + (image_noise_level*0.2)
    
            # posterior mean close to true value with some noise dependent bias
            posterior_mean = simulation_params_vector[0] + (np.random.randn()*image_noise_level*0.05)
    
            return norm(loc=posterior_mean,scale=posterior_std)
        elif self.simulation_params == 5:
            # simulate posterior mean slightly biased from true values
            bias = np.random.randn(self.simulation_params) * 0.05 
            posterior_mean = simulation_params_vector + bias

            # Simulate a 5x5 covariance matrix. A simple placeholder is a diagonal matrix, thus we are assuming no correlation between parameter uncertainties
            variances = np.full(self.simulation_params, 0.01) # base variance for each parameter
            posterior_cov = np.diag(variances)
    
            # return5d multivariate normal distribution object
            return multivariate_normal(mean=posterior_mean, cov=posterior_cov)

class Oracle:
    """Performs analysis pipeline to average performance over batch of images 
    on a given instrument design"""
    def __init__ (self,true_images,simulation_params,pipeline=NaiveNoiseInf):
        self.true_images = true_images
        self.simulation_params = simulation_params
        self.analysis_pipeline = pipeline(len(simulation_params.columns))

    def evaluate(self,design_param_x: float, batch_size=10) -> float:
        """
        An even more complex oracle with multiple local minima to challenge
        the acquisition functions.
        """
        scalar_design_param = design_param_x[0]
    
        batch_mses = []
        indices = np.random.choice(len(self.true_images), batch_size, replace=False)
        for i in indices:
            true_image = self.true_images[i][0]
            simulation_params_vector = self.simulation_params.iloc[i].values

            # simulate an observation w some noise
            noise = np.random.normal(0,scalar_design_param,true_image.shape)
            observed_image = true_image+noise

            # perform inference
            inferred_posterior = self.analysis_pipeline.infer_posterior(
                observed_image,
                true_image,
                simulation_params_vector
            )

            ## start calculating multidimensional mse
            # get posterior mean vector and covariance matrix
            posterior_mean = inferred_posterior.mean
            posterior_cov = inferred_posterior.cov

            # calculate 5element bias vector
            bias_vector = posterior_mean - simulation_params_vector

            # calculate outer product of bias vector (5x5 matrix)
            bias_term = np.outer(bias_vector,bias_vector)

            # mse matrix = bias term + covar term
            mse_matrix = bias_term + posterior_cov

            # final performance score = trace(mse matrix)
            batch_mses.append(np.trace(mse_matrix))
            
        performance_from_inference = np.mean(batch_mses)

        # design penalty with a "sweet spot" 
        optimal_noise_sweet_spot = 0.25 # Let's move the sweet spot
        
        # quadratic penalty
        main_valley = 0.05 * (scalar_design_param - optimal_noise_sweet_spot)**2
        
        #  sine wave to create ripples 
        ripples = -0.0005 * np.cos(50 * scalar_design_param)
        
        design_penalty = main_valley + ripples
        design_penalty = 0
        final_performance_y = performance_from_inference + design_penalty
        
        return final_performance_y
