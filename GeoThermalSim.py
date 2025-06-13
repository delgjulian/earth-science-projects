# GeoThermalSim.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_1d_heat_conduction(
    depth_model, dz, dt, total_time,
    thermal_conductivity_model, # k in W/(m·K)
    density_model,              # rho in kg/m^3
    specific_heat_model,        # cp in J/(kg·K)
    heat_production_model,      # A in W/m^3 (radiogenic heat production)
    initial_temperature_profile, # T_initial in K or °C
    top_boundary_temp,          # Surface temp in K or °C
    bottom_boundary_type='fixed_temp', # 'fixed_temp' or 'fixed_flux'
    bottom_boundary_value=None, # K or W/m^2 (value for the bottom boundary)
    display_animation=True,
    animation_skip_frames=100,  # Skip more frames for long simulations
    plot_snapshots_at_times=None # List of times (s) to plot specific snapshots
):
    """
    Simulates 1D transient heat conduction with internal heat production
    using the finite difference method (Explicit method).

    The governing equation is:
    rho * cp * d(T)/dt = d/dz(k * dT/dz) + A
    or in terms of thermal diffusivity (alpha = k / (rho * cp)):
    dT/dt = alpha * d^2(T)/dz^2 + A / (rho * cp)

    Parameters:
    -----------
    depth_model : float
        Total depth of the 1D model (m).
    dz : float
        Spatial grid spacing (m).
    dt : float
        Time step (s).
    total_time : float
        Total simulation time (s).
    thermal_conductivity_model : numpy.ndarray
        Array of thermal conductivities (W/(m·K)) for each depth point.
    density_model : numpy.ndarray
        Array of densities (kg/m^3) for each depth point.
    specific_heat_model : numpy.ndarray
        Array of specific heat capacities (J/(kg·K)) for each depth point.
    heat_production_model : numpy.ndarray
        Array of radiogenic heat production rates (W/m^3) for each depth point.
    initial_temperature_profile : numpy.ndarray
        Array of initial temperatures (K or °C) for each depth point.
    top_boundary_temp : float
        Fixed temperature at the top surface (z=0) (K or °C).
    bottom_boundary_type : str, optional
        Type of bottom boundary condition: 'fixed_temp' (fixed temperature)
        or 'fixed_flux' (fixed heat flux). Defaults to 'fixed_temp'.
    bottom_boundary_value : float, optional
        Value for the bottom boundary (K for 'fixed_temp', W/m^2 for 'fixed_flux').
        Defaults to None.
    display_animation : bool, optional
        If True, displays an animation of the temperature profile. Defaults to True.
    animation_skip_frames : int, optional
        Number of frames to skip in the animation for faster playback. Defaults to 100.
    plot_snapshots_at_times : list of floats, optional
        List of times (s) to plot specific snapshots.

    Returns:
    --------
    T_field : numpy.ndarray
        2D array (time x depth) of the temperature field.
    time_steps : numpy.ndarray
        Array of time steps used in the simulation.
    depth_coords : numpy.ndarray
        Array of depth coordinates.
    """

    # --- Setup Grid ---
    nz = int(depth_model / dz) + 1
    nt = int(total_time / dt) + 1
    depth_coords = np.linspace(0, depth_model, nz)
    time_steps = np.linspace(0, total_time, nt)

    # Validate model sizes
    if not (len(thermal_conductivity_model) == nz and
            len(density_model) == nz and
            len(specific_heat_model) == nz and
            len(heat_production_model) == nz and
            len(initial_temperature_profile) == nz):
        raise ValueError(f"All model arrays must have {nz} elements.")
    
    # Ensure models are numpy arrays
    k = np.array(thermal_conductivity_model)
    rho = np.array(density_model)
    cp = np.array(specific_heat_model)
    A = np.array(heat_production_model)

    # Calculate thermal diffusivity (alpha = k / (rho * cp))
    alpha = np.divide(k, (rho * cp), out=np.zeros_like(k), where=(rho*cp)!=0)

    # --- Initialize Temperature Field ---
    T_field = np.zeros((nt, nz))
    T_curr = np.array(initial_temperature_profile).copy()
    T_prev = np.array(initial_temperature_profile).copy()

    # Apply initial boundary conditions to T_curr
    T_curr[0] = top_boundary_temp
    if bottom_boundary_type == 'fixed_temp':
        T_curr[nz-1] = bottom_boundary_value

    # --- Stability Condition (for Explicit Finite Difference) ---
    alpha_max = np.max(alpha)
    stability_limit = dz**2 / (2 * alpha_max)
    if dt > stability_limit:
        print(f"WARNING: Explicit FD stability condition violated! dt ({dt:.2e}s) > dz^2/(2*alpha_max) ({stability_limit:.2e}s).")
        print("Simulation might be unstable. Consider reducing dt.")

    print(f"\nStarting 1D Heat Conduction Simulation ({nz} depth points, {nt} time steps)")
    print(f"Maximum Thermal Diffusivity: {alpha_max:.2e} m^2/s")
    print(f"Top boundary: {top_boundary_temp}°C (fixed)")
    print(f"Bottom boundary: {bottom_boundary_type} at {bottom_boundary_value} {('°C' if bottom_boundary_type == 'fixed_temp' else 'W/m^2')}")

    # --- Finite Difference Time Domain (FDTD) Loop ---
    for n in range(nt):
        T_prev[:] = T_curr[:]

        for i in range(1, nz - 1):
            d2T_dz2 = (T_prev[i+1] - 2 * T_prev[i] + T_prev[i-1]) / dz**2
            heat_prod_term = A[i] / (rho[i] * cp[i]) if (rho[i] * cp[i]) != 0 else 0
            T_curr[i] = T_prev[i] + dt * (alpha[i] * d2T_dz2 + heat_prod_term)

        # --- Apply Boundary Conditions for T_curr ---
        T_curr[0] = top_boundary_temp

        if bottom_boundary_type == 'fixed_temp':
            T_curr[nz-1] = bottom_boundary_value
        elif bottom_boundary_type == 'fixed_flux':
            T_curr[nz-1] = T_prev[nz-1] + dt * (alpha[nz-1] * (T_prev[nz-2] - T_prev[nz-1] - bottom_boundary_value * dz / k[nz-1]) / dz**2 \
                                              + A[nz-1] / (rho[nz-1] * cp[nz-1]))

        # Store temperature field snapshot
        T_field[n, :] = T_curr[:]

        if (n % (nt // 10)) == 0 and n > 0:
            print(f"  Time step {n}/{nt} ({n/nt*100:.0f}%)")

    print("Simulation complete.")

    # --- Animation (Optional) ---
    if display_animation:
        fig, ax = plt.subplots(figsize=(8, 10))
        line, = ax.plot(T_field[0, :], depth_coords, 'r-')
        
        ax.set_ylim(depth_model, 0)
        ax.set_xlim(np.min(initial_temperature_profile) - 20, np.max(initial_temperature_profile) + 100)

        ax.set_title("1D Heat Conduction", fontsize=14)
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Depth (m)", fontsize=12)
        ax.grid(True)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        def animate(frame):
            actual_frame = frame * animation_skip_frames
            if actual_frame >= nt:
                return line, time_text
            line.set_xdata(T_field[actual_frame, :])
            time_text.set_text(f'Time: {time_steps[actual_frame]/3.154e7:.2f} Myr')
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=nt // animation_skip_frames,
            interval=dt * animation_skip_frames * 1000, blit=True
        )
        plt.show()

    # --- Plot Specific Snapshots (Optional) ---
    if plot_snapshots_at_times:
        plt.figure(figsize=(8, 10))
        plt.title("Temperature Profiles at Specific Times", fontsize=14)
        plt.xlabel("Temperature (°C)", fontsize=12)
        plt.ylabel("Depth (m)", fontsize=12)
        plt.grid(True)
        plt.ylim(depth_model, 0)

        for target_time in plot_snapshots_at_times:
            time_idx = np.argmin(np.abs(time_steps - target_time))
            plt.plot(T_field[time_idx, :], depth_coords, label=f'{time_steps[time_idx]/3.154e7:.2f} Myr')
        
        plt.legend(fontsize=10)
        plt.show()


    return T_field, time_steps, depth_coords

def map_properties_to_layers(depth_coords, layer_depths, layer_properties):
    """
    Maps properties (k, rho, cp, A) to a layered Earth model.
    """
    nz = len(depth_coords)
    property_array = np.zeros(nz)

    for i in range(nz):
        current_depth = depth_coords[i]
        
        layer_idx = 0
        for depth_interface in layer_depths:
            if current_depth > depth_interface:
                layer_idx += 1
            else:
                break
        
        property_array[i] = layer_properties[layer_idx]
    
    return property_array


if __name__ == "__main__":
    # --- Define Simulation Parameters ---
    total_depth = 100000.0 # meters (100 km depth)
    dz = 100.0             # spatial grid spacing (m)
    
    # --- FIX: Significantly Increased dt for faster simulation ---
    dt = 3.154e7           # time step (s) - 10 years (was 1 year)
    
    # --- FIX: Reduced total_simulation_time for faster initial runs ---
    total_simulation_time_sec = 1e4 * 3.154e7 # Convert 10,000 years to seconds (was 50,000 years)

    # --- Define Geological Model (Layered Earth Properties) ---
    nz = int(total_depth / dz) + 1
    depth_coordinates_for_models = np.linspace(0, total_depth, nz)

    # Layer definitions (Crust and Mantle example)
    crust_mantle_interface_depth = 40000.0 # 40 km depth for crust-mantle boundary

    # Properties for each layer
    k_crust = 2.5       # W/(m·K)
    rho_crust = 2700.0  # kg/m^3
    cp_crust = 1000.0   # J/(kg·K)
    A_crust = 1.0e-6    # W/m^3 (typical radiogenic heat production in crust)

    k_mantle = 3.0      # W/(m·K)
    rho_mantle = 3300.0 # kg/m^3
    cp_mantle = 1200.0  # J/(kg·K)
    A_mantle = 0.05e-6  # W/m^3 (lower heat production in mantle)

    thermal_conductivity_array = map_properties_to_layers(
        depth_coordinates_for_models, [crust_mantle_interface_depth], [k_crust, k_mantle]
    )
    density_array = map_properties_to_layers(
        depth_coordinates_for_models, [crust_mantle_interface_depth], [rho_crust, rho_mantle]
    )
    specific_heat_array = map_properties_to_layers(
        depth_coordinates_for_models, [crust_mantle_interface_depth], [cp_crust, cp_mantle]
    )
    heat_production_array = map_properties_to_layers(
        depth_coordinates_for_models, [crust_mantle_interface_depth], [A_crust, A_mantle]
    )

    # --- Initial Temperature Profile (e.g., linear geothermal gradient) ---
    surface_temp_initial = 15.0 # °C
    initial_gradient = 0.02 # °C/m or K/m
    initial_temp_profile = surface_temp_initial + initial_gradient * depth_coordinates_for_models
    
    # --- Boundary Conditions ---
    top_bc_temp = 15.0 # °C (fixed surface temperature)
    
    bottom_bc_type = 'fixed_flux' # Options: 'fixed_temp', 'fixed_flux'
    bottom_bc_value = 0.03 # W/m^2 (for fixed_flux) or °C (for fixed_temp)

    # --- Plotting specific snapshots ---
    # Adjusted snapshot times to fit the new, shorter simulation time
    snapshot_times_myr = [0.001, 0.005, 0.01] # Myr (1,000, 5,000, 10,000 years)
    snapshot_times_sec = [t * 3.154e7 for t in snapshot_times_myr] # Convert to seconds


    # --- Run Simulation ---
    T_field_result, time_steps_result, depth_coords_result = simulate_1d_heat_conduction(
        depth_model=total_depth,
        dz=dz,
        dt=dt,
        total_time=total_simulation_time_sec,
        thermal_conductivity_model=thermal_conductivity_array,
        density_model=density_array,
        specific_heat_model=specific_heat_array,
        heat_production_model=heat_production_array,
        initial_temperature_profile=initial_temp_profile,
        top_boundary_temp=top_bc_temp,
        bottom_boundary_type=bottom_bc_type,
        bottom_boundary_value=bottom_bc_value,
        display_animation=True,
        animation_skip_frames=10, # Reduced skip frames for smoother animation
        plot_snapshots_at_times=snapshot_times_sec
    )

    print("\n1D Earth Thermal Conduction simulation complete. Check the generated plots and animation.")
