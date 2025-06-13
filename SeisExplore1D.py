# SeisExplore1D.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def ricker_wavelet(t, f_peak):
    """
    Generates a Ricker wavelet (second derivative of a Gaussian).

    Parameters:
    -----------
    t : numpy.ndarray
        Time array (s).
    f_peak : float
        Peak frequency of the wavelet (Hz).

    Returns:
    --------
    wavelet : numpy.ndarray
        Ricker wavelet amplitude.
    """
    # Ricker wavelet formula: (1 - 2*pi^2*f_peak^2*t^2) * exp(-pi^2*f_peak^2*t^2)
    r = (np.pi * f_peak * t)
    wavelet = (1 - 2 * r**2) * np.exp(-r**2)
    return wavelet

def calculate_acoustic_impedance(velocity_model, density_model):
    """
    Calculates the acoustic impedance (Z = rho * V) for each point in the model.

    Parameters:
    -----------
    velocity_model : numpy.ndarray
        Array of P-wave velocities (m/s).
    density_model : numpy.ndarray
        Array of densities (kg/m^3).

    Returns:
    --------
    acoustic_impedance : numpy.ndarray
        Array of acoustic impedance values (kg/m^2 s).
    """
    return density_model * velocity_model

def calculate_reflectivity_series(acoustic_impedance_model):
    """
    Calculates the reflection coefficients (reflectivity series) at each interface.
    Reflection coefficient R = (Z2 - Z1) / (Z2 + Z1)
    where Z1 is impedance of layer 1 and Z2 is impedance of layer 2.

    Parameters:
    -----------
    acoustic_impedance_model : numpy.ndarray
        Array of acoustic impedance values.

    Returns:
    --------
    reflectivity_series : numpy.ndarray
        Array of reflection coefficients. Length is (nx-1) for nx points.
    """
    reflectivity = np.zeros(len(acoustic_impedance_model) - 1)
    for i in range(len(acoustic_impedance_model) - 1):
        z1 = acoustic_impedance_model[i]
        z2 = acoustic_impedance_model[i+1]
        if (z1 + z2) != 0: # Avoid division by zero
            reflectivity[i] = (z2 - z1) / (z2 + z1)
    return reflectivity


def simulate_1d_seismic_wave(
    length_x, dx, dt, total_time,
    velocity_model, # Array of velocities for each layer
    density_model,  # Array of densities for each layer
    source_location_x, source_time, f_peak,
    boundary_type='absorbing',
    display_animation=True,
    animation_skip_frames=10,
    receiver_locations_x=None
):
    """
    Simulates 1D seismic (acoustic) wave propagation using the finite difference method.

    Parameters:
    -----------
    length_x : float
        Length of the 1D medium (m).
    dx : float
        Spatial grid spacing (m).
    dt : float
        Time step (s).
    total_time : float
        Total simulation time (s).
    velocity_model : numpy.ndarray
        Array of P-wave velocities (m/s) for each layer or grid point.
        Its size should correspond to the number of grid points (nx).
    density_model : numpy.ndarray
        Array of densities (kg/m^3) for each layer or grid point.
        Its size should correspond to the number of grid points (nx).
    source_location_x : float
        X-coordinate of the seismic source (m).
    source_time : float
        Time at which the source initiates (s).
    f_peak : float
        Peak frequency of the Ricker wavelet source (Hz).
    boundary_type : str, optional
        Type of boundary condition: 'absorbing' (simple 1st order absorbing)
        or 'rigid' (perfect reflection). Defaults to 'absorbing'.
    display_animation : bool, optional
        If True, displays an animation of the wavefield. Defaults to True.
    animation_skip_frames : int, optional
        Number of frames to skip in the animation for faster playback. Defaults to 10.
    receiver_locations_x : list of floats, optional
        List of x-coordinates (m) where synthetic seismograms will be recorded.

    Returns:
    --------
    u_field : numpy.ndarray
        2D array (time x space) of the displacement field.
    time_steps : numpy.ndarray
        Array of time steps used in the simulation.
    seismograms : dict
        Dictionary where keys are receiver x-coordinates and values are the
        recorded displacement vs. time.
    """

    # --- Setup Grid ---
    nx = int(length_x / dx) + 1
    nt = int(total_time / dt) + 1
    x_coords = np.linspace(0, length_x, nx)
    time_steps = np.linspace(0, total_time, nt)

    # Validate model sizes
    if len(velocity_model) != nx:
        raise ValueError(f"velocity_model must have {nx} elements, but has {len(velocity_model)}")
    if len(density_model) != nx:
        raise ValueError(f"density_model must have {nx} elements, but has {len(density_model)}")
    
    # Ensure models are numpy arrays
    v = np.array(velocity_model)
    rho = np.array(density_model)

    # --- Initialize Wavefield ---
    u_prev = np.zeros(nx)
    u_curr = np.zeros(nx)
    u_next = np.zeros(nx)

    # --- Courant-Friedrichs-Lewy (CFL) Condition Check ---
    cfl_limit = dx / np.max(v)
    if dt > cfl_limit:
        print(f"WARNING: CFL condition violated! dt ({dt:.2e}s) > dx/Vmax ({cfl_limit:.2e}s).")
        print("Simulation might be unstable. Consider reducing dt.")

    # --- Source Setup ---
    source_idx = int(round(source_location_x / dx))
    if not (0 <= source_idx < nx):
        raise ValueError(f"Source location {source_location_x}m is outside grid (0 to {length_x}m).")

    source_time_series = ricker_wavelet(time_steps - source_time, f_peak)
    
    # --- Seismogram Recording ---
    seismograms = {}
    receiver_indices = []
    if receiver_locations_x:
        for rx in receiver_locations_x:
            r_idx = int(round(rx / dx))
            if 0 <= r_idx < nx:
                receiver_indices.append(r_idx)
                seismograms[rx] = np.zeros(nt)
            else:
                print(f"Warning: Receiver at {rx}m is outside grid. Ignoring.")

    u_field = np.zeros((nt, nx)) # To store the entire wavefield over time

    print(f"\nStarting 1D Seismic Wave Simulation ({nx} spatial points, {nt} time steps)")
    print(f"Source at x={source_location_x}m, peak frequency={f_peak}Hz")
    print(f"Boundary type: {boundary_type}")

    # --- Finite Difference Time Domain (FDTD) Loop ---
    for n in range(nt):
        for i in range(1, nx - 1):
            # 1D Acoustic Wave Equation (with density variation)
            # rho * d^2u/dt^2 = d/dx(rho * v^2 * du/dx)
            
            rho_v2_plus_half = 0.5 * (rho[i] * v[i]**2 + rho[i+1] * v[i+1]**2)
            rho_v2_minus_half = 0.5 * (rho[i] * v[i]**2 + rho[i-1] * v[i-1]**2)

            term_right = (rho_v2_plus_half * (u_curr[i+1] - u_curr[i]) / dx - \
                          rho_v2_minus_half * (u_curr[i] - u_curr[i-1]) / dx) / dx

            u_next[i] = 2 * u_curr[i] - u_prev[i] + (dt**2 / rho[i]) * term_right

            # Inject source directly into the wavefield as displacement for clarity
            if i == source_idx:
                u_next[i] += source_time_series[n] * 1e-4 # Scale source amplitude for visibility

        # --- Boundary Conditions ---
        if boundary_type == 'rigid':
            u_next[0] = 0
            u_next[nx-1] = 0
        elif boundary_type == 'absorbing':
            u_next[0] = u_curr[0] + ((v[0] * dt / dx) * (u_curr[1] - u_curr[0]))
            u_next[nx-1] = u_curr[nx-1] + ((v[nx-1] * dt / dx) * (u_curr[nx-2] - u_curr[nx-1]))
        else:
            u_next[0] = 0
            u_next[nx-1] = 0

        # Update wavefields for next iteration
        u_prev[:] = u_curr[:]
        u_curr[:] = u_next[:]

        # Store wavefield snapshot
        u_field[n, :] = u_curr[:]

        # Record seismograms
        for idx, r_x in zip(receiver_indices, receiver_locations_x):
            seismograms[r_x][n] = u_curr[idx]

        if (n % (nt // 10)) == 0 and n > 0:
            print(f"  Time step {n}/{nt} ({n/nt*100:.0f}%)")

    print("Simulation complete.")

    # --- Animation (Optional) ---
    if display_animation:
        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot(x_coords, u_field[0, :], 'b-')
        ax.set_xlim(0, length_x)
        if np.max(np.abs(u_field)) > 1e-9:
            ax.set_ylim(-1.5 * np.max(np.abs(u_field)), 1.5 * np.max(np.abs(u_field)))
        else:
            ax.set_ylim(-1e-5, 1e-5) 

        ax.set_title("1D Seismic Wave Propagation")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Displacement")
        ax.grid(True)

        # Plot velocity model as a background
        ax_twin_v = ax.twinx()
        ax_twin_v.plot(x_coords, v, 'r--', alpha=0.5, label='Velocity (m/s)')
        ax_twin_v.set_ylabel("Velocity (m/s)", color='red')
        ax_twin_v.tick_params(axis='y', labelcolor='red')
        ax_twin_v.legend(loc='upper right')

        # Plot density model as another background
        ax_twin_rho = ax.twinx()
        ax_twin_rho.spines['right'].set_position(('outward', 60))
        ax_twin_rho.plot(x_coords, rho, 'g:', alpha=0.5, label='Density (kg/m³)')
        ax_twin_rho.set_ylabel("Density (kg/m³)", color='green')
        ax_twin_rho.tick_params(axis='y', labelcolor='green')
        ax_twin_rho.legend(loc='lower right')


        # Mark source and receivers
        ax.axvline(x=source_location_x, color='gray', linestyle=':', label='Source')
        if receiver_locations_x:
            for rx in receiver_locations_x:
                ax.axvline(x=rx, color='green', linestyle=':', label=f'Receiver at {rx}m')
        ax.legend(loc='upper left')


        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        def animate(frame):
            line.set_ydata(u_field[frame * animation_skip_frames, :])
            time_text.set_text(f'Time: {time_steps[frame * animation_skip_frames]:.3f} s')
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=nt // animation_skip_frames,
            interval=dt * animation_skip_frames * 1000, blit=True
        )
        plt.show()

    return u_field, time_steps, seismograms

def plot_seismograms(seismograms, time_steps, acoustic_impedance_model, reflectivity_series,
                     x_coords_global, noise_level=0.0, title="Synthetic Seismograms"):
    """
    Plots the synthetic seismograms recorded at receiver locations,
    along with acoustic impedance and reflectivity series.

    Parameters:
    -----------
    seismograms : dict
        Dictionary of seismograms (x-coord: displacement array).
    time_steps : numpy.ndarray
        Array of time steps.
    acoustic_impedance_model : numpy.ndarray
        Array of acoustic impedance values.
    reflectivity_series : numpy.ndarray
        Array of reflection coefficients.
    x_coords_global : numpy.ndarray
        Array of global x-coordinates for spatial plotting.
    noise_level : float, optional
        Standard deviation of Gaussian noise to add to each seismogram (mGal).
        Defaults to 0.0 (no noise).
    title : str, optional
        Title for the plot. Defaults to "Synthetic Seismograms".
    """
    if not seismograms:
        print("No seismograms to plot.")
        return

    sorted_receivers = sorted(seismograms.keys())

    # Create a figure with 3 subplots: Impedance, Reflectivity, Seismograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), gridspec_kw={'width_ratios': [1, 0.5, 4]})

    # --- Plot 1: Acoustic Impedance Profile ---
    ax0 = axes[0]
    ax0.plot(acoustic_impedance_model, x_coords_global, 'k-', linewidth=2)
    ax0.set_title("Acoustic Impedance", fontsize=14)
    ax0.set_xlabel("Impedance (kg/m²s)", fontsize=12)
    ax0.set_ylabel("Distance (m)", fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.invert_yaxis() # Invert y-axis to represent depth/distance correctly
    # Set y-limits to match the extent of the model
    ax0.set_ylim(x_coords_global[-1], x_coords_global[0]) 

    # --- Plot 2: Reflectivity Series ---
    ax1 = axes[1]
    # Plot reflectivity at the location of interfaces
    # Reflectivity series has length nx-1, corresponds to interfaces between points
    # We plot it against the midpoints of the dx intervals or just x_coords_global[:-1]
    interface_x_coords = x_coords_global[:-1] + dx / 2.0
    ax1.vlines(0, x_coords_global[0], x_coords_global[-1], colors='gray', linestyles='dotted') # Zero line
    ax1.plot(reflectivity_series, interface_x_coords, 'r-o', markersize=4, linewidth=1.5)
    ax1.set_title("Reflectivity Series", fontsize=14)
    ax1.set_xlabel("Reflection Coefficient", fontsize=12)
    ax1.set_ylabel("Distance (m)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_yaxis() # Invert y-axis
    ax1.set_ylim(x_coords_global[-1], x_coords_global[0])


    # --- Plot 3: Synthetic Seismograms (Shot Gather) ---
    ax2 = axes[2]
    max_amplitude = np.max(np.abs(list(seismograms.values())))
    if max_amplitude == 0:
        max_amplitude = 1e-9

    # Offset scaling: make it such that traces don't overlap too much.
    offset_per_trace = 2.0 * max_amplitude # Adjust this to control vertical spacing

    for i, rx in enumerate(sorted_receivers):
        trace = seismograms[rx].copy() # Use .copy() to avoid modifying original array
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.max(np.abs(trace)), trace.shape)
            trace += noise

        # Offset each trace
        offset_trace = trace + i * offset_per_trace
        ax2.plot(time_steps, offset_trace, 'k-', linewidth=1.0) # Black line for trace

        # Fill positive wiggles
        ax2.fill_between(time_steps, i * offset_per_trace, offset_trace, where=trace > 0, color='blue', alpha=0.5)
        
        # Add receiver label (at a fixed time or relative to actual arrival time)
        ax2.text(time_steps[-1] * 1.02, i * offset_per_trace, f'{rx}m', va='center', fontsize=9)

    ax2.set_title(title, fontsize=14)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Receiver Distance (m)", fontsize=12) # Changed label to be more descriptive
    ax2.set_yticks([i * offset_per_trace for i in range(len(sorted_receivers))]) # Set Y-ticks to trace baselines
    ax2.set_yticklabels([f'{rx}m' for rx in sorted_receivers]) # Use receiver distances as Y-tick labels
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, time_steps[-1])
    # Set y-limits to encompass all traces
    ax2.set_ylim(-offset_per_trace, len(sorted_receivers) * offset_per_trace)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Define Simulation Parameters ---
    x_length = 500.0  # meters (length of the 1D medium)
    dx = 1.0         # Spatial grid spacing (m)
    # --- CRITICAL FIX: REDUCED DT FOR CFL STABILITY ---
    dt = 0.0002      # Time step (s) - crucial for stability (CFL condition)!
    total_sim_time = 0.5 # seconds (total duration of the simulation)

    # --- Define Geological Model (Velocity & Density Layers) ---
    nx = int(x_length / dx) + 1 # Number of spatial grid points
    
    # Global x-coordinates for mapping layer properties
    x_coords_global = np.linspace(0, x_length, nx) 

    # Scenario: Three-Layer Medium
    velocity_profile = np.zeros(nx)
    density_profile = np.zeros(nx)

    # Layer 1: Top Layer (e.g., unconsolidated sediments)
    layer1_end_x = 150.0  # meters (depth of first interface)
    layer1_velocity = 1500.0 # m/s
    layer1_density = 1800.0 # kg/m^3

    # Layer 2: Middle Layer (e.g., sandstone)
    layer2_end_x = 350.0 # meters (depth of second interface)
    layer2_velocity = 2500.0 # m/s
    layer2_density = 2200.0 # kg/m^3

    # Layer 3: Bottom Layer (e.g., shale or bedrock)
    layer3_velocity = 3500.0 # m/s
    layer3_density = 2600.0 # kg/m^3

    # Populate velocity and density models based on layer boundaries
    for i in range(nx):
        if x_coords_global[i] <= layer1_end_x:
            velocity_profile[i] = layer1_velocity
            density_profile[i] = layer1_density
        elif x_coords_global[i] <= layer2_end_x:
            velocity_profile[i] = layer2_velocity
            density_profile[i] = layer2_density
        else:
            velocity_profile[i] = layer3_velocity
            density_profile[i] = layer3_density
    
    # --- Source Parameters ---
    source_x_location = 50.0 # meters (location of the seismic source)
    source_t_start = 0.03   # seconds (small delay before the wavelet starts)
    peak_frequency = 25.0    # Hz (controls the dominant frequency of the seismic wave)

    # --- Boundary Conditions ---
    boundary_condition_type = 'absorbing'

    # --- Receiver Locations ---
    # Place receivers at various distances to observe reflections.
    receiver_xs = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0]

    # --- Plotting Parameters ---
    seismogram_noise_level = 0.1 # Standard deviation of noise relative to max trace amplitude (0.0 to 1.0 recommended)

    # --- Run Simulation ---
    u_field_result, time_steps_result, seismograms_result = simulate_1d_seismic_wave(
        length_x=x_length,
        dx=dx,
        dt=dt,
        total_time=total_sim_time,
        velocity_model=velocity_profile, # Pass the defined velocity array
        density_model=density_profile,   # Pass the defined density array
        source_location_x=source_x_location,
        source_time=source_t_start,
        f_peak=peak_frequency,
        boundary_type=boundary_condition_type,
        display_animation=True,
        animation_skip_frames=10,
        receiver_locations_x=receiver_xs
    )

    # --- Calculate Acoustic Impedance and Reflectivity ---
    acoustic_impedance = calculate_acoustic_impedance(velocity_profile, density_profile)
    reflectivity = calculate_reflectivity_series(acoustic_impedance)

    # --- Plot Seismograms with AI and Reflectivity ---
    plot_seismograms(seismograms_result, time_steps_result,
                     acoustic_impedance, reflectivity, x_coords_global,
                     noise_level=seismogram_noise_level)

    print("\n1D Seismic wave simulation and analysis complete. Check the generated plots and animation.")
