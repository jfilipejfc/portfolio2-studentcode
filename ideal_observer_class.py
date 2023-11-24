# -*- coding: utf-8 -*-
"""
Implements an ideal Bayesian observer model for causal inference.

Estimates whether visual and auditory cues have a common cause or separate 
causes. Also estimates the respective source locations for the stimuli.

Based on the probabilistic model described in Kording et al. (2007),
section "Material and Methods" , subsections "Estimating the probability
of a common cause" and "Optimally estimating the position".

Kording, K.P., Beierholm, U., Ma, W.J., Quartz, S., Tenenbaum, J.B., Shams, 
L.: Causal Inference in Multisensory Perception. PLoS ONE 2(9), e943 (2007),
doi:10.1371/journal.pone.0000943

First released on Thu Nov 22 2023  

@author: João Filipe Ferreira
"""

# Global imports
import numpy as np  # Maths

class IdealObserver:
    """
    Ideal Bayesian observer model for causal inference regarding visual and 
    auditory stimuli, as per Kording et al. (2007).
    
    Attributes:
        _mu_p: Mean of prior distribution over source locations 
        _sigma_p: Standard deviation of prior over source locations
        _p_common: Prior probability that stimuli have a common cause
        _sigma_v: Standard deviation of visual sensory noise distribution
        _sigma_a: Standard deviation of auditory sensory noise distribution 
    """
    def __init__(self, p_common, mu_p, sigma_p, sigma_v, sigma_a):
        """
        Initializes the ideal observer with parameters of the probabilistic 
        model.
        """
 
        # Define starting parameters
        self._mu_p     = mu_p       # prior mean
        self._sigma_p  = sigma_p    # prior std
        self._p_common = p_common   # probability of perceiving a common cause for visual and auditory stimuli
        self._sigma_v  = sigma_v    # auditory std
        self._sigma_a  = sigma_a    # auditory std
        pass

    def _calculate_likelihood_single_source(self, x_v, x_a):
        """
        Calculates the likelihood of the visual and auditory cues corresponding to a single source,
        as per Kording et al. (2007), eq (4).

        Args:
            x_v: Visual reading for experimental trial  
            x_a: Auditory reading for experimental trial
       
        Returns:
            l_single_source: likelihood for readings given single source scenario 
        """

        # Get overall sigma
        sigma =  np.sqrt((self._sigma_v**2)*(self._sigma_a**2)+(self._sigma_v**2)*(self._sigma_p**2)+(self._sigma_a**2)*(self._sigma_p**2))

        # Get overall "mu numerator"
        mu_numerator = (x_v-x_a)**2*(self._sigma_p)**2+(x_v-self._mu_p)**2*(self._sigma_a)**2+(x_a-self._mu_p)**2*(self._sigma_v)**2

        # Compute likelihood probability
        l_single_source = 1 / (2 * np.pi * sigma) * np.exp(-mu_numerator / (2 * sigma**2))

        return l_single_source

    def _calculate_likelihood_separate_sources(self, x_v, x_a):      
        """
        Calculates the likelihood of the visual and auditory cues corresponding to a separate sources,
        as per Kording et al. (2007), eq (6).

        Args:
            x_v: Visual reading for experimental trial  
            x_a: Auditory reading for experimental trial
       
        Returns:
            l_separate_sources: likelihood for readings given single source scenario 
        """ 
        # Get overall sigma
        sigma =  np.sqrt(((self._sigma_v**2)+(self._sigma_p**2))*((self._sigma_a**2)+(self._sigma_p**2)))

        # Get main expression for exponential
        exponential_expr = ((x_v-self._mu_p)**2/(self._sigma_v**2+self._sigma_p**2))+((x_a-self._mu_p)**2/(self._sigma_a**2+self._sigma_p**2))

        # Compute likelihood probability
        l_separate_sources = 1 / (2 * np.pi * sigma) * np.exp((-1/2) * exponential_expr)

        return l_separate_sources


    def _calculate_p_single_source(self, x_v, x_a):
        """
        Calculates the posterior probability of a single source given visual and auditory cues,
        p_single_source, as per Kording et al. (2007), eq (2).

        The posterior probability of separate sources is, therefore, 1 - p_single_source
        
        Args:
            x_v: Visual reading for experimental trial  
            x_a: Auditory reading for experimental trial
       
        Returns:
            p_single_source: posterior probability of a single source given visual and auditory cues 
        """ 

        # Calculate the probability of the visual and auditory cues corresponding to a single source
        # (i.e. the "ventriloquist effect")

        # Calculate the likelihood of the visual and auditory cues corresponding to a single source
        l_c_1 = self._calculate_likelihood_single_source(x_v, x_a)

        # Calculate the likelihood of the visual and auditory cues corresponding to separate sources    
        l_c_2 = self._calculate_likelihood_separate_sources(x_v, x_a)

        # Calculate probability of the "ventriloquist effect"
        p_single_source = l_c_1*self._p_common/(l_c_1*self._p_common + l_c_2*(1-self._p_common))

        return p_single_source

    def _calculate_v_estimate_params(self, x_v):
        """
        Calculates ideal visual estimate parameters (i.e. assuming separate sources),
        as per Kording et al. (2007), eq (11), left.
        
        Args:
            x_v: Visual reading for experimental trial  
       
        Returns:
            s_v, sigma_est_v: ideal visual estimate parameters (i.e. assuming separate sources) 
        """  
        s_v = (x_v / self._sigma_v**2 + self._mu_p / self._sigma_p**2) / (1 / self._sigma_v**2 + 1 / self._sigma_p**2)
        sigma_est_v = 1 / self._sigma_v**2 + 1 / self._sigma_p**2

        return s_v, sigma_est_v

    def _calculate_a_estimate_params(self, x_a):
        """
        Calculates ideal auditory estimate parameters (i.e. assuming separate sources),
        as per Kording et al. (2007), eq (11), left.
         
        Args:
            x_a: Auditory reading for experimental trial  
       
        Returns:
            s_a, sigma_est_a: ideal auditory estimate parameters (i.e. assuming separate sources) 
        """  
        s_a = (x_a / self._sigma_a**2 + self._mu_p / self._sigma_p**2) / (1 / self._sigma_a**2 + 1 / self._sigma_p**2)
        sigma_est_a = 1 / self._sigma_a**2 + 1 / self._sigma_p**2

        return s_a, sigma_est_a    

    def _calculate_joint_va_estimate_params(self, x_v, x_a):
        """
        Calculates ideal visuo-auditory estimate (i.e. assuming single source),
        as per Kording et al. (2007), eq (12).
        
        Args:
            x_v: Visual reading for experimental trial  
            x_a: Auditory reading for experimental trial
       
        Returns:
            s_va: ideal visuo-auditory estimate (i.e. assuming single source) 
        """   
        # Calculate joint visuo-auditory estimate
        s_va = (x_v / self._sigma_v**2 + x_a / self._sigma_a**2 + self._mu_p / self._sigma_p**2) / (1 / self._sigma_v**2 + 1 / self._sigma_a**2 + 1 / self._sigma_p**2)

        return s_va

    def calculate_estimates(self, x_v, x_a):
        """
        Estimates whether stimuli arise from common cause or separate causes,
        as per Kording et al. (2007), eq (2).
        
        Also estimates respective source locations for visual and auditory
        stimuli, as per Kording et al. (2007), eqs (9-11).
        
        Args:
            x_v: Visual reading for experimental trial  
            x_a: Auditory reading for experimental trial
       
        Returns:
            s_v, s_a, p_single_source, s_v_est, s_a_est:
                Estimates and posterior parameters for the trial  
        """

        # Calculate the probability of the visual and auditory cues corresponding to a single source
        # (i.e. the "ventriloquist effect")
        p_single_source = self._calculate_p_single_source(x_v, x_a)

        # Calculate separate visual ideal estimate parameters
        s_v, _ = self._calculate_v_estimate_params(x_v)

        # Calculate separate auditory ideal estimate parameters
        s_a, _ = self._calculate_v_estimate_params(x_a)
    
        # Calculate joint visuo-auditory ideal estimate
        s_va = self._calculate_joint_va_estimate_params(x_v, x_a)
    
        # Calculate visual and auditory estimates given estimate C
        s_v_est            = p_single_source*s_va + (1-p_single_source)*s_v # eq (9)
        s_a_est            = p_single_source*s_va + (1-p_single_source)*s_a # eq (10)

        return s_v, s_a, p_single_source, s_v_est, s_a_est