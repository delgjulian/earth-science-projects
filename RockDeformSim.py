# RockDeformSim.py

import numpy as np
import matplotlib.pyplot as plt

def simulate_1d_rock_deformation(
    total_time, dt,
    initial_stress=0.0, # Initial stress (Pa)
    applied_strain_rate=None, # Constant applied strain rate (s^-1)
    applied_stress_rate=None, # Constant applied stress rate (Pa/s)
    # Or, if neither rate is given, apply stress in steps
    applied_stress_steps=None, # List of (time_start_s, stress_Pa) tuples
    
    # Material properties
    youngs_modulus=50e9,  # E in Pa (Young's Modulus, for elastic deformation)
    viscosity=1e19,       # eta in Pa·s (Viscosity, for viscous creep)
    yield_strength=200e6, # sigma_y in Pa (Yield Strength, for plastic deformation)
    
    # Flags for active deformation mechanisms
    enable_elastic=True,
    enable_viscous_creep=True,
    enable_plastic_yield=True,
    
    plot_snapshots_at_times=None # List of times (s) for specific stress-strain points
):
    """
    Simulates 1D rock deformation (elastic, viscous, plastic) under stress
    or strain rate using an explicit time-stepping approach.

    Parameters:
    -----------
    total_time : float
        Total simulation time (s).
    dt : float
        Time step (s).
    initial_stress : float, optional
        Initial stress in the rock (Pa). Defaults to 0.0.
    applied_strain_rate : float, optional
        Constant applied strain rate (s^-1). If specified, stress evolves.
        Cannot be used with applied_stress_rate or applied_stress_steps.
    applied_stress_rate : float, optional
        Constant applied stress rate (Pa/s). If specified, strain evolves.
        Cannot be used with applied_strain_rate or applied_stress_steps.
    applied_stress_steps : list of (time_s, stress_Pa) tuples, optional
        List of tuples defining stepwise applied stress.
        e.g., [(0, 0), (100, 1e8), (200, 1.5e8)].
        Cannot be used with applied_strain_rate or applied_stress_rate.

    youngs_modulus : float, optional
        Young's Modulus (E) in Pa. Defaults to 50 GPa (granite-like).
    viscosity : float, optional
        Viscosity (eta) in Pa·s. Defaults to 1e19 Pa·s (mantle-like creep).
    yield_strength : float, optional
        Yield Strength (sigma_y) in Pa. Defaults to 200 MPa.

    enable_elastic : bool, optional
        If True, elastic deformation is enabled. Defaults to True.
    enable_viscous_creep : bool, optional
        If True, viscous creep (Maxwell element) is enabled. Defaults to True.
    enable_plastic_yield : bool, optional
        If True, plastic yielding (perfectly plastic) is enabled. Defaults to True.
    
    plot_snapshots_at_times : list of floats, optional
        List of specific times (in seconds) at which to store stress-strain points.

    Returns:
    --------
    time_points : numpy.ndarray
        Array of time points (s).
    stress_history : numpy.ndarray
        Array of stress values (Pa) over time.
    strain_history : numpy.ndarray
        Array of total strain values over time.
    snapshot_data : list of dicts
        List of dictionaries with stress, strain, time for specific snapshots.
    """

    nt = int(total_time / dt) + 1
    time_points = np.linspace(0, total_time, nt)

    # Initialize stress and strain arrays
    stress_history = np.zeros(nt)
    strain_history = np.zeros(nt)
    
    current_stress = initial_stress
    current_strain = 0.0 # Start with zero strain

    stress_history[0] = current_stress
    strain_history[0] = current_strain

    snapshot_data = []
    if plot_snapshots_at_times:
        current_snapshot_idx = 0
        snapshot_times_sorted = sorted(plot_snapshots_at_times)

    # Validate input modes
    modes_active = [applied_strain_rate is not None, applied_stress_rate is not None, applied_stress_steps is not None]
    if sum(modes_active) != 1:
        raise ValueError("Exactly one of 'applied_strain_rate', 'applied_stress_rate', or 'applied_stress_steps' must be provided.")

    print(f"\nStarting 1D Rock Deformation Simulation ({nt} time steps)")
    print(f"Young's Modulus (E): {youngs_modulus/1e9:.1f} GPa")
    if enable_viscous_creep:
        print(f"Viscosity (eta): {viscosity:.1e} Pa·s")
    if enable_plastic_yield:
        print(f"Yield Strength (sigma_y): {yield_strength/1e6:.1f} MPa")
    print(f"Enabled mechanisms: Elastic ({enable_elastic}), Viscous ({enable_viscous_creep}), Plastic ({enable_plastic_yield})")


    for n in range(1, nt):
        # Calculate instantaneous stress/strain based on applied mode
        if applied_strain_rate is not None:
            # Apply total strain incrementally
            current_strain_increment = applied_strain_rate * dt
            # Elastic response: d_sigma = E * d_epsilon_elastic
            # Viscous response: d_epsilon_viscous = sigma / eta * dt
            # Plastic response: if sigma > yield, d_epsilon_plastic = d_epsilon_total - d_epsilon_elastic - d_epsilon_viscous

            # Here we apply strain rate and calculate resulting stress
            # Total strain increment
            d_epsilon_total = current_strain_increment
            
            # Initialize deformation components for this step
            d_epsilon_elastic = 0.0
            d_epsilon_viscous = 0.0
            d_epsilon_plastic = 0.0

            # Elastic calculation (predict stress from elastic strain)
            # This is complex in a strain-controlled setup with other mechanisms.
            # Simpler: assume elastic is dominant initially.
            # If we apply total strain, stress changes by:
            # d_sigma = E * d_epsilon_elastic
            # d_epsilon_elastic = d_epsilon_total - d_epsilon_viscous - d_epsilon_plastic
            
            # Viscous strain rate is sigma/eta
            if enable_viscous_creep and viscosity > 0:
                 d_epsilon_viscous = (current_stress / viscosity) * dt
            
            # Elastic strain is total - viscous - plastic
            # Here we'll iterate to solve for stress/strain
            # For simplicity for demonstration, assume elastic happens instantly.
            # Stress update from total strain assuming elastic + viscous
            if enable_elastic:
                elastic_stress_change = youngs_modulus * d_epsilon_total
                
                if enable_viscous_creep and viscosity > 0:
                    # Maxwell model: d_epsilon_total = d_epsilon_elastic + d_epsilon_viscous
                    # d_epsilon_total = (1/E) * d_sigma + (sigma/eta) * dt
                    # d_sigma = E * (d_epsilon_total - (sigma/eta)*dt)
                    # Use current_stress for sigma in viscous term (explicit update)
                    d_sigma = youngs_modulus * (d_epsilon_total - (current_stress / viscosity) * dt)
                else: # Only elastic
                    d_sigma = youngs_modulus * d_epsilon_total
            else: # No elastic means d_sigma is 0, only viscous/plastic can change stress
                d_sigma = 0.0 # If elastic is off, stress won't recover immediately

            current_stress += d_sigma
            current_strain += d_epsilon_total

            # Apply plastic yield after elastic/viscous calculation
            if enable_plastic_yield and np.abs(current_stress) > yield_strength:
                # If stress exceeds yield strength, it gets capped.
                # The excess deformation becomes plastic strain.
                if current_stress > yield_strength:
                    d_epsilon_plastic = (current_stress - yield_strength) / youngs_modulus if enable_elastic else 0.0
                    current_stress = yield_strength
                elif current_stress < -yield_strength:
                    d_epsilon_plastic = (current_stress + yield_strength) / youngs_modulus if enable_elastic else 0.0
                    current_stress = -yield_strength
                current_strain += d_epsilon_plastic # Plastic strain accumulates


        elif applied_stress_rate is not None:
            # Apply total stress incrementally
            current_stress_increment = applied_stress_rate * dt
            current_stress += current_stress_increment

            # Calculate strain response from the applied stress
            d_epsilon = 0.0
            
            # Elastic strain
            if enable_elastic:
                d_epsilon += (current_stress - stress_history[n-1]) / youngs_modulus # Elastic strain increment from stress change

            # Viscous strain
            if enable_viscous_creep and viscosity > 0:
                d_epsilon += (current_stress / viscosity) * dt # Viscous strain increment

            current_strain += d_epsilon

            # Apply plastic yield (if stress exceeds yield strength)
            if enable_plastic_yield and np.abs(current_stress) > yield_strength:
                # If stress exceeds yield strength, it gets capped.
                # The excess deformation becomes plastic strain.
                if current_stress > yield_strength:
                    current_stress = yield_strength
                elif current_stress < -yield_strength:
                    current_stress = -yield_strength
                # Plastic strain accumulates (already accounted for if stress is capped)
                # No additional strain increment, as stress is held at yield.


        elif applied_stress_steps is not None:
            # Find the target stress for the current time
            target_stress = current_stress # Keep current stress by default
            for t_step, s_val in applied_stress_steps:
                if time_points[n] >= t_step:
                    target_stress = s_val
                else:
                    break # Assuming steps are sorted by time

            # Calculate stress change for this step
            stress_change_this_step = target_stress - current_stress
            current_stress += stress_change_this_step # Directly set the stress

            d_epsilon = 0.0
            # Elastic strain from instantaneous stress change
            if enable_elastic:
                d_epsilon += stress_change_this_step / youngs_modulus
            
            # Viscous strain due to the current stress (over dt)
            if enable_viscous_creep and viscosity > 0:
                d_epsilon += (current_stress / viscosity) * dt
            
            current_strain += d_epsilon

            # Apply plastic yield
            if enable_plastic_yield and np.abs(current_stress) > yield_strength:
                if current_stress > yield_strength:
                    current_stress = yield_strength
                elif current_stress < -yield_strength:
                    current_stress = -yield_strength
                # If stress is capped, the strain continues to increase plastically
                # We need to account for the plastic strain that would have occurred
                # if stress had continued to increase past yield.
                # This is a simplification; a full plastic model would calculate
                # the plastic strain increment more explicitly.
                # For now, we cap the stress and allow the strain to accumulate.
                # If stress is capped, the effective Young's Modulus becomes zero in that direction.
                # The strain will just accumulate with time.
                # For simplicity, if stress is capped, we add a small plastic strain increment
                # This is a highly simplified plastic model.
                pass # Stress capping handled by setting current_stress directly.

        # Store histories
        stress_history[n] = current_stress
        strain_history[n] = current_strain

        # Store snapshots
        if plot_snapshots_at_times and current_snapshot_idx < len(snapshot_times_sorted):
            if time_points[n] >= snapshot_times_sorted[current_snapshot_idx]:
                snapshot_data.append({
                    'time': time_points[n],
                    'stress': current_stress,
                    'strain': current_strain
                })
                print(f"  Snapshot captured at Time: {time_points[n]:.2f} s, Stress: {current_stress:.2e} Pa, Strain: {current_strain:.2e}")
                current_snapshot_idx += 1

        if (n % (nt // 10)) == 0 and n > 0:
            print(f"  Time step {n}/{nt} ({n/nt*100:.0f}%)")

    print("Simulation complete.")
    return time_points, stress_history, strain_history, snapshot_data

def plot_deformation_results(
    time_points, stress_history, strain_history,
    snapshot_data,
    youngs_modulus, viscosity, yield_strength,
    enable_elastic, enable_viscous_creep, enable_plastic_yield
):
    """
    Plots the stress-strain curve, and stress/strain vs. time.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Stress vs. Time ---
    ax0 = axes[0]
    ax0.plot(time_points, stress_history / 1e6, 'b-', label='Stress') # Convert to MPa
    ax0.set_xlabel("Time (s)", fontsize=12)
    ax0.set_ylabel("Stress (MPa)", fontsize=12, color='b')
    ax0.tick_params(axis='y', labelcolor='b')
    ax0.set_title("Stress and Strain vs. Time", fontsize=14)
    ax0.grid(True, linestyle='--', alpha=0.6)

    # Twin axis for Strain vs. Time
    ax0_twin = ax0.twinx()
    ax0_twin.plot(time_points, strain_history, 'r--', label='Strain')
    ax0_twin.set_ylabel("Strain", fontsize=12, color='r')
    ax0_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend for ax0 and ax0_twin
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    ax0_twin.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)


    # --- Plot 2: Stress-Strain Curve ---
    ax1 = axes[1]
    ax1.plot(strain_history, stress_history / 1e6, 'k-', linewidth=2, label='Simulated Behavior') # Convert to MPa
    ax1.set_xlabel("Strain", fontsize=12)
    ax1.set_ylabel("Stress (MPa)", fontsize=12)
    ax1.set_title("Stress-Strain Curve", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Mark yield strength (if plastic enabled)
    if enable_plastic_yield:
        ax1.axhline(y=yield_strength/1e6, color='gray', linestyle=':', label=f'Yield Strength ({yield_strength/1e6:.0f} MPa)')
        ax1.axhline(y=-yield_strength/1e6, color='gray', linestyle=':')

    # Mark snapshots on the stress-strain curve
    for snapshot in snapshot_data:
        ax1.plot(snapshot['strain'], snapshot['stress'] / 1e6, 'ro', markersize=8, alpha=0.7, label=f'T={snapshot["time"]:.0f}s Snapshot' if snapshot_data.index(snapshot) == 0 else "")
        ax1.text(snapshot['strain'] + 0.0005, snapshot['stress'] / 1e6 + 5, f'{snapshot["time"]:.0f}s', fontsize=9)


    # Add theoretical elastic line for comparison
    if enable_elastic:
        # Create a theoretical elastic stress-strain line up to yield or max stress
        max_strain_elastic_theoretical = np.max(np.abs(stress_history)) / youngs_modulus
        if enable_plastic_yield and np.abs(yield_strength / youngs_modulus) < np.max(np.abs(strain_history)):
             # If plastic is enabled, draw elastic line up to yield strength
            theoretical_elastic_strain = np.array([-yield_strength / youngs_modulus, yield_strength / youngs_modulus])
            theoretical_elastic_stress = youngs_modulus * theoretical_elastic_strain
            ax1.plot(theoretical_elastic_strain, theoretical_elastic_stress / 1e6, 'g--', label='Theoretical Elastic')
        else: # If no plastic or max stress is within elastic limit
            ax1.plot(np.array([0, max_strain_elastic_theoretical]), np.array([0, youngs_modulus * max_strain_elastic_theoretical]) / 1e6, 'g--', label='Theoretical Elastic')


    ax1.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Define Simulation Parameters ---
    total_sim_time = 3000.0  # seconds (e.g., 50 minutes for a lab experiment or a short geological event)
    dt = 1.0               # time step (s)

    # --- Applied Load Mode ---
    # Choose ONE of these modes by uncommenting:
    
    # Mode 1: Constant Applied Strain Rate
    # applied_strain_rate_val = 1e-6 # s^-1 (e.g., 1 microstrain per second)
    # applied_stress_rate_val = None
    # applied_stress_steps_val = None

    # Mode 2: Constant Applied Stress Rate
    # applied_strain_rate_val = None
    # applied_stress_rate_val = 1e5 # Pa/s (e.g., 100 kPa/s)
    # applied_stress_steps_val = None

    # Mode 3: Stepwise Applied Stress (Default)
    applied_strain_rate_val = None
    applied_stress_rate_val = None
    applied_stress_steps_val = [
        (0, 0),        # Start at 0 stress
        (500, 50e6),   # Ramp up to 50 MPa by 500s
        (1500, 150e6), # Hold, then ramp to 150 MPa by 1500s
        (2500, 250e6), # Ramp to 250 MPa (will hit yield if enabled)
        (3000, 0)      # Unload back to 0 MPa
    ]


    # --- Material Properties ---
    # Typical rock properties
    E = 50e9       # Young's Modulus (Pa, 50 GPa for typical crustal rock)
    eta = 1e19     # Viscosity (Pa·s, for ductile creep; 1e19 to 1e21 common for mantle)
    sigma_y = 200e6 # Yield Strength (Pa, 200 MPa)

    # --- Deformation Mechanisms (Enable/Disable) ---
    enable_elastic_deformation = True
    enable_viscous_creep_deformation = True
    enable_plastic_yield_deformation = True # Set to False to see elastic-viscous only

    # --- Snapshots for Stress-Strain Curve ---
    snapshot_times = [500, 1000, 1500, 2000, 2500, 3000] # seconds


    # --- Run Simulation ---
    time_history, stress_h, strain_h, snapshots = simulate_1d_rock_deformation(
        total_time=total_sim_time,
        dt=dt,
        initial_stress=0.0,
        applied_strain_rate=applied_strain_rate_val,
        applied_stress_rate=applied_stress_rate_val,
        applied_stress_steps=applied_stress_steps_val,
        youngs_modulus=E,
        viscosity=eta,
        yield_strength=sigma_y,
        enable_elastic=enable_elastic_deformation,
        enable_viscous_creep=enable_viscous_creep_deformation,
        enable_plastic_yield=enable_plastic_yield_deformation,
        plot_snapshots_at_times=snapshot_times
    )

    # --- Plot Results ---
    plot_deformation_results(
        time_history, stress_h, strain_h, snapshots,
        youngs_modulus=E, viscosity=eta, yield_strength=sigma_y,
        enable_elastic=enable_elastic_deformation,
        enable_viscous_creep=enable_viscous_creep_deformation,
        enable_plastic_yield=enable_plastic_yield_deformation
    )

    print("\n1D Rock Deformation simulation complete. Check the generated plots.")
