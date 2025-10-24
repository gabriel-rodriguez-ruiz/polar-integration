#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:17:02 2025

@author: gabriel
"""

import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize_scalar
import time

class NumericalFunctionMinimizer:
    def __init__(self, numerical_function, bounds, max_evaluations=1000):
        """
        numerical_function: function that takes x and returns f(x) through hard computation
        bounds: tuple (min, max) defining search domain
        max_evaluations: maximum number of function evaluations
        """
        self.func = numerical_function
        self.bounds = bounds
        self.max_evals = max_evaluations
        self.evaluation_count = 0
        self.history = []
    
    def wrapped_function(self, x):
        """Wrapper to track evaluations and handle numerical issues"""
        if self.evaluation_count >= self.max_evals:
            raise RuntimeError("Maximum evaluations reached")
        
        self.evaluation_count += 1
        
        try:
            result = self.func(x)
            self.history.append((x, result))
            return result
        except Exception as e:
            # Return a large value for failed evaluations
            print(f"Evaluation failed at x={x}: {e}")
            return float('inf')
        from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    
    def bayesian_optimization(self, n_calls=100, random_starts=10):
        """Bayesian optimization using Gaussian Processes"""
        space = [Real(self.bounds[0], self.bounds[1], name='x')]
        
        @use_named_args(space)
        def objective(**params):
            return self.wrapped_function(params['x'])
        
        result = gp_minimize(
            objective, space, 
            n_calls=n_calls,
            n_initial_points=random_starts,
            random_state=42
        )
        
        return result.x[0], result.fun
    

# Example of a computationally expensive function
def hard_numerical_function(x):
    """
    Example of a function that requires significant computation.
    This could be a numerical integration, PDE solution, etc.
    """
    # Simulate expensive computation
    time.sleep(0.01)  # 10ms delay to simulate hard computation
    
    # Example complex function with multiple minima
    result = (x - 2)**2 * np.sin(5*x) + 0.1*(x + 3)**2
    result += 0.5 * np.exp(-0.5*((x - 5)/0.5)**2)  # Additional local minimum
    
    # Add some noise to simulate numerical instability
    result += 0.01 * np.random.normal()
    
    return result

# Usage example
def main():
    # Define search bounds
    bounds = (-5, 8)
    
    # Create minimizer
    minimizer = NumericalFunctionMinimizer(
        hard_numerical_function, 
        bounds, 
        max_evaluations=500
    )
    
    # Run comprehensive optimization
    best_x, best_fx, all_results = minimizer.comprehensive_global_minimization()
    
    print(f"\nOverall best result:")
    print(f"x = {best_x:.6f}, f(x) = {best_fx:.6f}")
    print(f"Total function evaluations: {minimizer.evaluation_count}")
    
    # Plot results if desired
    if len(minimizer.history) > 0:
        import matplotlib.pyplot as plt
        
        history_x = [point[0] for point in minimizer.history]
        history_fx = [point[1] for point in minimizer.history]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(history_x, history_fx, alpha=0.6, c=range(len(history_x)), cmap='viridis')
        plt.colorbar(label='Evaluation order')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Function evaluations')
        
        plt.subplot(1, 2, 2)
        plt.semilogy(history_fx)
        plt.xlabel('Evaluation number')
        plt.ylabel('f(x)')
        plt.title('Convergence history')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()