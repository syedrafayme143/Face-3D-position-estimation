"""
Linear Kalman Filter Implementation
For state estimation in linear systems with Gaussian noise
"""

from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class LinKalman:
    """
    Linear Kalman Filter for state estimation.
    
    Implements the standard Kalman filter equations for linear systems
    with Gaussian process and measurement noise.
    """
    
    def __init__(self, x, A, H, R, Q):
        """
        Initialize the Kalman filter.
        
        Args:
            x (np.ndarray): Initial state estimate (n x 1) or (n,)
            A (np.ndarray): State transition matrix (n x n)
            H (np.ndarray): Observation/measurement matrix (m x n)
            R (np.ndarray): Measurement noise covariance (m x m)
            Q (np.ndarray): Process noise covariance (n x n)
        """
        self.x = x.flatten() if x.ndim > 1 else x  # Ensure 1D array
        self.A = A
        self.H = H
        self.R = R
        self.Q = Q
        self.P = Q.copy()  # Initialize state covariance with process noise
    
    def predictState(self):
        """
        Predict the next state using the state transition model.
        
        Prediction equations:
            x_{t|t-1} = A * x_{t-1|t-1}
            P_{t|t-1} = A * P_{t-1|t-1} * A^T + Q
        
        Returns:
            tuple: (x_predicted, P_predicted)
                - x_predicted (np.ndarray): Predicted state (n,)
                - P_predicted (np.ndarray): Predicted covariance (n x n)
        """
        x_predicted = np.matmul(self.A, self.x)
        P_predicted = np.matmul(self.A, np.matmul(self.P, self.A.T)) + self.Q
        
        return (x_predicted, P_predicted)
    
    def predictMeasurement(self):
        """
        Predict the measurement based on current state.
        
        Equation:
            z_{t|t-1} = H * x_{t|t-1}
        
        Returns:
            np.ndarray: Predicted measurement (m,)
        """
        z_predicted = np.matmul(self.H, self.x)
        return z_predicted.flatten()
    
    def computeKalmanGain(self):
        """
        Compute the Kalman gain matrix.
        
        Equation:
            K = P_{t|t-1} * H^T * (H * P_{t|t-1} * H^T + R)^{-1}
        
        Returns:
            np.ndarray: Kalman gain matrix K (n x m)
        """
        x_predicted, P_predicted = self.predictState()
        
        # Compute intermediate matrices
        PHT = np.matmul(P_predicted, self.H.T)           # P * H^T
        HPHT = np.matmul(self.H, PHT)                    # H * P * H^T
        S = HPHT + self.R                                 # Innovation covariance
        S_inv = np.linalg.inv(S)                         # (H * P * H^T + R)^{-1}
        
        K = np.matmul(PHT, S_inv)                        # Kalman gain
        
        return K
    
    def update(self, z):
        """
        Update the state estimate with a new measurement.
        
        Update equations:
            K = P_{t|t-1} * H^T * (H * P_{t|t-1} * H^T + R)^{-1}
            x_{t|t} = x_{t|t-1} + K * (z_t - H * x_{t|t-1})
            P_{t|t} = P_{t|t-1} - K * H * P_{t|t-1}
        
        Args:
            z (np.ndarray): Measurement vector (m x 1) or (m,)
        
        Returns:
            tuple: (x_updated, P_updated)
                - x_updated (np.ndarray): Updated state estimate (n,)
                - P_updated (np.ndarray): Updated covariance (n x n)
        """
        # Prediction step
        x_predicted, P_predicted = self.predictState()
        z_predicted = self.predictMeasurement()
        
        # Compute Kalman gain
        K = self.computeKalmanGain()
        
        # Innovation (measurement residual)
        innovation = z.flatten() - z_predicted
        
        # Update state estimate
        self.x = x_predicted + np.matmul(K, innovation)
        self.x = self.x.flatten()  # Ensure 1D array
        
        # Update covariance estimate
        self.P = P_predicted - np.matmul(K, np.matmul(self.H, P_predicted))
        
        return self.x, self.P


if __name__ == "__main__":
    """
    Test the Kalman filter with a simple 1D constant position model.
    
    System:
        - True state: constant at 20
        - Measurements: noisy observations with mean=20, std=1
        - Process noise: very small (Q=0.01)
        - Measurement noise: R=1
    """
    
    # Initialize filter parameters
    x = np.array([0.0])           # Initial state estimate (starting at 0)
    A = np.array([[1.0]])         # State transition (constant position)
    H = np.array([[1.0]])         # Measurement model (direct observation)
    R = np.array([[1.0]])         # Measurement noise covariance
    Q = np.array([[0.01]])        # Process noise covariance
    z_mean = 20.0                 # True measurement mean
    
    print("=" * 60)
    print("Kalman Filter Test - 1D Constant Position Model")
    print("=" * 60)
    print(f"Initial state x: {x}")
    print(f"State transition A:\n{A}")
    print(f"Measurement model H:\n{H}")
    print(f"Measurement noise R:\n{R}")
    print(f"Process noise Q:\n{Q}")
    print(f"True measurement mean: {z_mean}")
    print("=" * 60)
    
    # Create Kalman filter
    kalman = LinKalman(x, A, H, R, Q)
    
    # Storage for plotting
    x_arr = []        # State estimates
    P_arr = []        # Covariance estimates
    z_arr = []        # Measurements
    step_arr = []     # Time steps
    
    # Run filter for 100 steps
    num_steps = 100
    print(f"\nRunning filter for {num_steps} steps...\n")
    
    for step in range(num_steps):
        # Store current estimates
        x_arr.append(kalman.x[0])
        P_arr.append(kalman.P[0, 0])
        step_arr.append(step)
        
        # Print detailed info for first few steps
        if step < 5 or step % 20 == 0:
            print(f"Step {step}:")
            print(f"  Predict State: {kalman.predictState()}")
            print(f"  Predict Measurement: {kalman.predictMeasurement()}")
            print(f"  Kalman Gain K: {kalman.computeKalmanGain()}")
        
        # Generate noisy measurement
        z = np.array([np.random.normal(z_mean, np.sqrt(R[0, 0]))])
        z_arr.append(z[0])
        
        # Update filter with measurement
        x_updated, P_updated = kalman.update(z)
        
        if step < 5 or step % 20 == 0:
            print(f"  Measurement z: {z[0]:.2f}")
            print(f"  Updated state: {x_updated[0]:.2f}, Covariance: {P_updated[0, 0]:.4f}")
            print()
    
    # Print final results
    print("=" * 60)
    print("Final Results:")
    print(f"  Final state estimate: {kalman.x[0]:.2f}")
    print(f"  Final covariance: {kalman.P[0, 0]:.4f}")
    print(f"  True value: {z_mean}")
    print(f"  Error: {abs(kalman.x[0] - z_mean):.2f}")
    print("=" * 60)
    
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: State estimate vs measurements
    axs[0].plot(step_arr, x_arr, 'b-', linewidth=2, label='Kalman Estimate')
    axs[0].plot(step_arr, z_arr, 'rx', markersize=4, alpha=0.5, label='Measurements')
    axs[0].axhline(y=z_mean, color='g', linestyle='--', linewidth=1.5, label='True Value')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Kalman Filter: State Estimate vs Noisy Measurements')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Covariance over time
    axs[1].plot(step_arr, P_arr, 'b-', linewidth=2)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Covariance P')
    axs[1].set_title('State Covariance Over Time (Uncertainty)')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nPlots displayed. The Kalman filter should:")
    print("  1. Start at initial estimate (0)")
    print("  2. Converge toward true value (20)")
    print("  3. Smooth out measurement noise")
    print("  4. Reduce covariance (uncertainty) over time")
