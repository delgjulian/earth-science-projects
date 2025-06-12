# GravityAnomaly2D.py

import numpy as np
import matplotlib.pyplot as plt

def calculate_gravity_anomaly_2d_prism(
    x_profile, x1, x2, z1, z2, density_contrast
):
    """
    Calculates the vertical gravity anomaly (in mGal) along a 2D profile
    caused by a 2D horizontal rectangular prism (infinite in y-direction).

    The formula used is derived from integrating the gravitational effect of a 2D prism:
    Δgz(x) = 2 * G * Δρ * [ (x - x_edge) * ln((x - x_edge)^2 + z^2)^(1/2) - z * arctan((x - x_edge) / z) ]

    This formula is applied for each corner of the prism (x1, z1), (x2, z1), (x1, z2), (x2, z2)
    with alternating signs.

    Parameters:
    -----------
    x_profile : numpy.ndarray
        Array of x-coordinates (m) along which the gravity anomaly is calculated.
    x1 : float
        Left edge of the prism (m).
    x2 : float
        Right edge of the prism (m).
    z1 : float
        Top depth of the prism (m). Note: Depths are positive downwards.
    z2 : float
        Bottom depth of the prism (m). Note: z2 > z1.
    density_contrast : float
        Density contrast of the prism relative to the surrounding medium (kg/m^3).
        Positive for denser body, negative for less dense body.

    Returns:
    --------
    delta_g : numpy.ndarray
        Array of vertical gravity anomaly values (mGal) corresponding to x_profile.
    """

    # Gravitational constant G in m^3 kg^-1 s^-2 (6.674 × 10^-11)
    # Convert to mGal for typical geophysical units: 1 mGal = 10^-5 m/s^2
    # So, G_mGal = G_SI / 10^-5 = 6.674e-11 / 1e-5 = 6.674e-6
    G = 6.674e-11 # SI units (m^3 kg^-1 s^-2)
    G_mGal = G * 1e5 # Conversion factor to mGal (multiply by 10^5)

    delta_g = np.zeros_like(x_profile, dtype=float)

    # Define the corners of the prism for the calculation signs
    # (x_edge, z_depth, sign)
    corners = [
        (x1, z1, 1),   # Top-left corner (positive contribution)
        (x2, z1, -1),  # Top-right corner (negative contribution)
        (x1, z2, -1),  # Bottom-left corner (negative contribution)
        (x2, z2, 1)    # Bottom-right corner (positive contribution)
    ]

    for x_obs in x_profile:
        anomaly_at_point = 0.0
        for x_c, z_c, sign in corners:
            # Shift x-coordinate to be relative to the corner
            x_prime = x_obs - x_c

            # Handle potential division by zero if z_c is zero (at surface)
            # and x_prime is also zero. This is usually avoided by keeping z_c > 0.
            # However, arctan(inf) is pi/2, so if z_c is very small, this approaches.
            
            # Avoid issues with log(0) for x_prime = 0, z_c = 0 which shouldn't happen with z > 0.
            # Handle the specific case where z_c is zero or very close to it.
            # For practical purposes in geophysics, z1 is usually > 0.
            if z_c == 0: # If the depth is exactly zero, it's a special case (line source)
                # This function assumes z > 0, so if z_c is 0, this approximation is less accurate.
                # In real models, prisms are typically buried, so z1 > 0.
                term_z = 0.0 # Contribution of z*arctan(x/z) term approaches 0 as z->0 for x!=0
            else:
                term_z = z_c * np.arctan(x_prime / z_c) # np.arctan handles inf correctly

            # Calculate the term for a single corner
            # x * ln(sqrt(x^2 + z^2)) - z * arctan(x/z)
            # Note: np.log(0) gives -inf, so check for x_prime=0 and z_c=0
            # For our current setup, z_c should always be > 0.
            term_x = x_prime * np.log(np.sqrt(x_prime**2 + z_c**2)) if (x_prime**2 + z_c**2) > 0 else 0.0
            
            anomaly_at_point += sign * (term_x - term_z)
        
        # Multiply by 2 * G_mGal * density_contrast
        delta_g[x_profile == x_obs] = 2 * G_mGal * density_contrast * anomaly_at_point

    return delta_g


def plot_gravity_model_and_anomaly(
    x_profile, gravity_anomaly, x1, x2, z1, z2, density_contrast,
    noise_level=0.0, show_model=True
):
    """
    Plots the 2D geological model (prism) and the calculated gravity anomaly.

    Parameters:
    -----------
    x_profile : numpy.ndarray
        Array of x-coordinates for the profile.
    gravity_anomaly : numpy.ndarray
        Array of gravity anomaly values (mGal).
    x1, x2, z1, z2 : float
        Coordinates defining the prism (left, right, top_depth, bottom_depth).
    density_contrast : float
        Density contrast of the prism (kg/m^3).
    noise_level : float, optional
        Standard deviation of Gaussian noise to add to the anomaly (mGal).
        Defaults to 0.0 (no noise).
    show_model : bool, optional
        Whether to show the conceptual geological model. Defaults to True.
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # --- Plot Geological Model ---
    if show_model:
        ax0 = axes[0]
        # Draw the prism
        prism_color = 'lightcoral' if density_contrast > 0 else 'lightskyblue'
        ax0.fill([x1, x2, x2, x1, x1], [z1, z1, z2, z2, z1], color=prism_color, alpha=0.7)
        ax0.plot([x1, x2, x2, x1, x1], [z1, z1, z2, z2, z1], color='black', linewidth=1)
        
        # Labels and title
        ax0.set_title("Conceptual Geological Model (2D Prism)", fontsize=14)
        ax0.set_xlabel("Horizontal Distance (m)", fontsize=12)
        ax0.set_ylabel("Depth (m)", fontsize=12)
        ax0.set_xlim(x_profile.min(), x_profile.max())
        ax0.set_ylim(z2 + 0.1 * (z2 - z1), 0) # Inverted y-axis for depth
        ax0.invert_yaxis()
        ax0.grid(True, linestyle='--', alpha=0.6)
        
        # Add text for density contrast
        ax0.text( (x1+x2)/2, (z1+z2)/2, f"Δρ = {density_contrast:.1f} kg/m³",
                 color='black', ha='center', va='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))
        
        # Draw ground surface
        ax0.axhline(y=0, color='brown', linewidth=2, label='Ground Surface')
        ax0.legend()

    # --- Plot Gravity Anomaly ---
    ax1 = axes[1]
    
    # Theoretical anomaly
    ax1.plot(x_profile, gravity_anomaly, 'b-', linewidth=2, label='Theoretical Anomaly')

    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, gravity_anomaly.shape)
        noisy_anomaly = gravity_anomaly + noise
        ax1.plot(x_profile, noisy_anomaly, 'r.', markersize=5, alpha=0.7, label='Noisy Data')

    ax1.set_title("Vertical Gravity Anomaly Profile", fontsize=14)
    ax1.set_xlabel("Horizontal Distance (m)", fontsize=12)
    ax1.set_ylabel("Gravity Anomaly (mGal)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    ax1.set_xlim(x_profile.min(), x_profile.max())
    ax1.set_ylim(gravity_anomaly.min() * 1.2, gravity_anomaly.max() * 1.2) # Auto-scale Y with some padding
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Define Geological Model Parameters ---
    # Profile dimensions
    profile_length = 500.0  # meters
    num_points = 101       # Number of measurement points along the profile
    x_coords = np.linspace(-profile_length/2, profile_length/2, num_points)

    # Prism parameters
    prism_x1 = -50.0      # Left horizontal edge (m)
    prism_x2 = 50.0       # Right horizontal edge (m)
    prism_z1 = 20.0       # Top depth (m, positive downwards)
    prism_z2 = 80.0       # Bottom depth (m, positive downwards)
    
    # Density contrast (kg/m^3)
    # Common values: basalt (2900) vs granite (2650) => ~250 kg/m3
    # Ore body (5000) vs host rock (2700) => ~2300 kg/m3
    # Salt dome (2100) vs sediments (2400) => -300 kg/m3
    delta_rho = 300.0 # kg/m^3 (e.g., a denser intrusion)

    # --- Simulation Parameters ---
    noise_standard_deviation = 0.5 # mGal (e.g., typical noise in field measurements)
    show_geological_model = True

    # --- Calculate Gravity Anomaly ---
    theoretical_anomaly = calculate_gravity_anomaly_2d_prism(
        x_coords, prism_x1, prism_x2, prism_z1, prism_z2, delta_rho
    )

    # --- Plot Results ---
    plot_gravity_model_and_anomaly(
        x_coords, theoretical_anomaly, prism_x1, prism_x2, prism_z1, prism_z2,
        delta_rho, noise_level=noise_standard_deviation,
        show_model=show_geological_model
    )

    print("\nGravity anomaly simulation complete. Check the generated plots.")
