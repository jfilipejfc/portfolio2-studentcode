# -*- coding: utf-8 -*-
"""
Illustration of the "ventriloquist effect" using generative model and
an ideal Bayesian observer model, inspired on Kording et al. (2007) and
also described in Ferreira & Dias (2014).

A generative model is used to simulate experimental trial data, which models visual
and auditory stimuli corresponding to either one source or two independent sources.

An ideal Bayesian observer is then used to implement a model to estimate whether stimuli arise 
from a common cause ("ventriloquist effect") and to estimate individual stimulus source locations
from the generated experimental data.

Finally, the performance of the ideal observer is evaluated for the simulated experiment
w/ multiple trials.

Kording, K.P., Beierholm, U., Ma, W.J., Quartz, S., Tenenbaum, J.B., Shams, 
L.: Causal Inference in Multisensory Perception. PLoS ONE 2(9), e943 (2007),
doi:10.1371/journal.pone.0000943

João Filipe Ferreira & Jorge Dias, Probabilistic Approaches for Robotic Perception,
Springer International Publishing, 2014, Chapter 4.

First released on Thu Nov 22 2023

@author: João Filipe Ferreira  
"""

# Global imports
import numpy as np  # Maths

# Local plotter function imports
from plotters import plot_estimates
from plotters import plot_simple_performance_metrics

# Local ideal observer class import
from ideal_observer_class import IdealObserver

# Local experiment simulation setup function import
from experiment_setup import experiment_simulation

def main():
    """
    Runs the experiment simulation and analyses ideal observer performance.
    
    Simulates experimental trials generating visual and auditory stimuli 
    samples. Feeds these into an ideal observer model to estimate if stimuli  
    share a common cause and to estimate individual stimulus source locations
    accordingly. Plots estimates and posterior distributions when number of trials
    is small. 
    
    Evaluates the ideal observer's performance at the end of the experiment and
    plots metrics summarising performance over all trials.
    """

    # Define shared generative model and ideal observer model parameters
    mu_p      = 0     # source distribution prior mean         (degrees)
    sigma_p   = 40.0  # source distribution prior std          (degrees)
    p_common  = 0.28  # prior for perceiving a common cause for visual and auditory stimuli
    sigma_v   = 2.14  # Standard deviation for vision models   (degrees)
    sigma_a   = 9.2   # Standard deviation for auditory models (degrees)

    # Define experimental conditions
    trials   = 10    # number of trials in the experiment
    p_com_exp = 0.5     # (uniform) prior for generating a common cause for visual and auditory stimuli
                        # in experiment

    # Use generative model to simulate experiment trial data
    real_C_exp, real_s_v_exp, real_s_a_exp, x_v_exp, x_a_exp = \
        experiment_simulation(p_com_exp, mu_p, sigma_p, sigma_v, sigma_a, trials)

    # Create ideal observer object w/ relevant init data
    ideal_observer = IdealObserver(p_common, mu_p, sigma_p, sigma_v, sigma_a)

    # Initialise ideal observer performance metrics
    conf_mat_C  = [[0, 0],
                   [0, 0]]           # Confusion matrix for C
    mae_v = 0                        # Mean absolute error for v estimates
    mae_a = 0                        # Mean absolute error for a estimates
    mas_v = 0                        # Mean absolute shift for v estimates
    mas_a = 0                        # Mean absolute shift for a estimates

    # Cycle through experimental trials and respective simulation data and analyse ideal observer performance
    for real_C, real_s_v, real_s_a, x_v, x_a in zip(real_C_exp, real_s_v_exp, real_s_a_exp, x_v_exp, x_a_exp):

        # Get ideal observer estimates given current trial sensor readings
        s_v, s_a, p_single_source, s_v_est, s_a_est = ideal_observer.calculate_estimates(x_v, x_a)

        if trials <= 10: # Not to plot if experiment has a lot of trials!
            # Plot distributions
            plot_estimates(mu_p, sigma_p, s_v, sigma_v, s_a, sigma_a, # prior and partials
                        p_single_source, s_v_est, s_a_est)         # estimates

        # Contribution to confusion matrix for C
        est_C = int(p_single_source <= 0.5) + 1 # est_C = estimation for C given p_single_source

        if est_C == real_C: # in case estimate for C matches real C for trial
            if real_C == 1:  # in case of single source match
                conf_mat_C[0][0] += 1
            else:           # in case of 2 independent sources match
                conf_mat_C[1][1] += 1

        else: # in case estimate for C doesn't match real C for trial
            if real_C == 1:  # in case of single source mismatch
                conf_mat_C[0][1] += 1
            else:            # in case of 2 independent sources mismatch
                conf_mat_C[1][0] += 1

        # Angular estimation errors for s_v_est & s_a_est
        mae_v += abs(s_v_est - real_s_v)
        mae_a += abs(s_a_est - real_s_a)

        # Shift from ideal estimates to actual estimates
        mas_v += abs(s_v - s_v_est)
        mas_a += abs(s_a - s_a_est)

    # Compute final metrics values by averaging by number of trials
    mae_v /= trials
    mae_a /= trials
    mas_v /= trials
    mas_a /= trials
    conf_mat_C = np.array(conf_mat_C)
    conf_mat_C = np.divide(conf_mat_C, float(trials/100))

    # Show results
    plot_simple_performance_metrics(conf_mat_C, mae_v, mae_a, mas_v, mas_a)

    # ** STUDENT: ADD ANY OPTIONAL METRICS HERE **



if __name__ == "__main__":
    # Run main function
    main()
