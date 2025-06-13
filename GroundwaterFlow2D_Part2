# GroundwaterFlow2D.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def simulate_2d_groundwater_flow(
    length_x, length_y, dx, dy,
    hydraulic_conductivity_field, # K in m/s (2D array)
    source_sink_field=None,       # Q in m^3/s (2D array, positive for source, negative for sink)
    
    # Boundary Conditions (fixed_head or no_flow)
    # Each can be a fixed value (float) or a 1D array along the boundary.
    bc_left_type='fixed_head', bc_left_value=10.0,
    bc_right_type='fixed_head', bc_right_value=5.0,
    bc_top_type='no_flow', bc_top_value=None, # No_flow does not need a value
    bc_bottom_type='no_flow', bc_bottom_value=None,

    tolerance=1e-5,               # Convergence tolerance for hydraulic head (m)
    max_iterations=10000,         # Maximum number of iterations for solver
    initial_head_field=None       # Initial guess for head (2D array)
):
    """
    Simulates 2D steady-state groundwater flow using the Finite Difference Method (FDM).
    Solves the Laplace equation for hydraulic head (h):
    d/dx(Kx * dh/dx) + d/dy(Ky * dh/dy) + Q = 0
    Assuming isotropic K for simplicity (Kx = Ky = K).

    Parameters:
    -----------
    length_x, length_y : float
        Dimensions of the domain (m).
    dx, dy : float
        Spatial grid spacing (m).
    hydraulic_conductivity_field : numpy.ndarray (2D)
        Array of hydraulic conductivity values (K in m/s) for each grid cell.
    source_sink_field : numpy.ndarray (2D), optional
        Array of source/sink terms (Q in m^3/s per unit area) for each grid cell.
        Positive for injection, negative for pumping. Defaults to None (no sources/sinks).
    bc_left_type, bc_right_type, bc_top_type, bc_bottom_type : str
        Type of boundary condition for each side: 'fixed_head' (Dirichlet) or 'no_flow' (Neumann).
    bc_left_value, bc_right_value, bc_top_value, bc_bottom_value : float or numpy.ndarray, optional
        Value for fixed_head boundaries (head in m). For no_flow, set to None.
        If array, must match the length of the boundary.
    tolerance : float, optional
        Convergence tolerance for the hydraulic head (m). Defaults to 1e-5.
    max_iterations : int, optional
        Maximum number of iterations for the iterative solver. Defaults to 10000.
    initial_head_field : numpy.ndarray (2D), optional
        Initial guess for the hydraulic head field. If None, initializes to 0.

    Returns:
    --------
    h_field : numpy.ndarray (2D)
        Final converged hydraulic head field (m).
    vx_field, vy_field : numpy.ndarray (2D)
        Darcy velocities in X and Y directions (m/s).
    x_coords, y_coords : numpy.ndarray
        Arrays of x and y coordinates.
    """

    # --- Setup Grid ---
    nx = int(length_x / dx) + 1
    ny = int(length_y / dy) + 1
    x_coords = np.linspace(0, length_x, nx)
    y_coords = np.linspace(0, length_y, ny)

    # Validate K field size
    if hydraulic_conductivity_field.shape != (ny, nx):
        raise ValueError(f"hydraulic_conductivity_field must have shape ({ny}, {nx}).")
    K_field = np.array(hydraulic_conductivity_field)

    # Validate Source/Sink field size
    if source_sink_field is None:
        Q_field = np.zeros((ny, nx))
    else:
        if source_sink_field.shape != (ny, nx):
            raise ValueError(f"source_sink_field must have shape ({ny}, {nx}).")
        Q_field = np.array(source_sink_field)

    # --- Initialize Hydraulic Head ---
    if initial_head_field is None:
        h_curr = np.zeros((ny, nx))
    else:
        if initial_head_field.shape != (ny, nx):
            raise ValueError(f"initial_head_field must have shape ({ny}, {nx}).")
        h_curr = np.array(initial_head_field).copy()

    h_prev = h_curr.copy()

    print(f"\nStarting 2D Steady-State Groundwater Flow Simulation ({nx}x{ny} spatial points)")
    print(f"Tolerance: {tolerance}, Max Iterations: {max_iterations}")

    # --- Iterative Solver (Gauss-Seidel like update) ---
    # We iterate until the change in head is below tolerance or max_iterations is reached.
    for iteration in range(max_iterations):
        for j in range(ny): # Loop over Y-direction (rows)
            for i in range(nx): # Loop over X-direction (columns)
                
                # --- Apply Boundary Conditions first (overwrite internal updates if BC applies) ---
                if i == 0: # Left boundary
                    if bc_left_type == 'fixed_head':
                        h_curr[j, i] = bc_left_value if isinstance(bc_left_value, (int, float)) else bc_left_value[j]
                        continue # Skip internal update for fixed boundary
                if i == nx - 1: # Right boundary
                    if bc_right_type == 'fixed_head':
                        h_curr[j, i] = bc_right_value if isinstance(bc_right_value, (int, float)) else bc_right_value[j]
                        continue
                if j == 0: # Bottom boundary (y=0 in lower origin)
                    if bc_bottom_type == 'fixed_head':
                        h_curr[j, i] = bc_bottom_value if isinstance(bc_bottom_value, (int, float)) else bc_bottom_value[i]
                        continue
                if j == ny - 1: # Top boundary (y=length_y in lower origin)
                    if bc_top_type == 'fixed_head':
                        h_curr[j, i] = bc_top_value if isinstance(bc_top_value, (int, float)) else bc_top_value[i]
                        continue

                # --- Finite Difference Equation for Internal Nodes ---
                # Based on the node's neighbors and K-values at interfaces.
                # The equation is simplified for isotropic K and uniform grid.
                # K_avg_right = 0.5 * (K_field[j,i] + K_field[j, i+1])
                # This formulation (using K at cell centers) makes the derivatives at half-steps slightly tricky.
                # A simpler approach (for homogeneous K) is: h_i,j = (h_i+1,j + h_i-1,j + h_i,j+1 + h_i,j-1) / 4
                # For heterogeneous K, it's weighted average:

                # Coefficients based on conductance (K / dx or K / dy)
                # Left, Right, Bottom, Top neighbors (for node i,j)
                coeff_xp = 0.5 * (K_field[j, i] + K_field[j, min(i+1, nx-1)]) / dx**2 # K at i+1/2,j
                coeff_xn = 0.5 * (K_field[j, i] + K_field[j, max(i-1, 0)]) / dx**2   # K at i-1/2,j
                coeff_yp = 0.5 * (K_field[j, i] + K_field[min(j+1, ny-1), i]) / dy**2 # K at j+1/2,i
                coeff_yn = 0.5 * (K_field[j, i] + K_field[max(j-1, 0), i]) / dy**2   # K at j-1/2,i
                
                # Check for boundary conditions for neighbors if they are fixed_head or no_flow
                # No-flow boundary implies K * dh/dn = 0, which means derivative at boundary is 0.
                # This is handled by setting the 'ghost' node equal to the internal node's value.
                # In an explicit update, this means skipping the update for the boundary node or
                # modifying the stencil.
                # For this iterative solver, we can manage no_flow by treating the boundary as reflective.

                # Sum of coefficients for the central node
                sum_coeffs = coeff_xp + coeff_xn + coeff_yp + coeff_yn

                # Handle no-flow boundaries (reflect gradient)
                h_xp, h_xn, h_yp, h_yn = h_prev[j,i], h_prev[j,i], h_prev[j,i], h_prev[j,i] # Default to self if no flow

                # Neighbors' heads
                if i + 1 < nx: h_xp = h_prev[j, i+1]
                else: # Right boundary
                    if bc_right_type == 'no_flow': h_xp = h_prev[j, i]
                    # if fixed_head, handled by setting h_curr at the boundary

                if i - 1 >= 0: h_xn = h_prev[j, i-1]
                else: # Left boundary
                    if bc_left_type == 'no_flow': h_xn = h_prev[j, i]

                if j + 1 < ny: h_yp = h_prev[j+1, i]
                else: # Top boundary
                    if bc_top_type == 'no_flow': h_yp = h_prev[j, i]

                if j - 1 >= 0: h_yn = h_prev[j-1, i]
                else: # Bottom boundary
                    if bc_bottom_type == 'no_flow': h_yn = h_prev[j, i]

                # Update equation
                # Sum of (Coeff * Neighbor_Head) - Q_source_sink * dx * dy
                # h_curr[j,i] = ( (coeff_xp * h_xp + coeff_xn * h_xn + coeff_yp * h_yp + coeff_yn * h_yn) - Q_field[j,i] ) / sum_coeffs
                # Re-deriving for explicit update based on common FDM for steady-state diffusion-like equation:
                # h_i,j = (A*h_i+1,j + B*h_i-1,j + C*h_i,j+1 + D*h_i,j-1 + F) / (A+B+C+D)
                
                # Simplified update for iteration (Gauss-Seidel like, using latest available h_curr values)
                h_curr[j,i] = (
                    coeff_xp * h_curr[j, min(i+1, nx-1)] + # Use updated h_curr if available (Gauss-Seidel)
                    coeff_xn * h_curr[j, max(i-1, 0)] +
                    coeff_yp * h_curr[min(j+1, ny-1), i] +
                    coeff_yn * h_curr[max(j-1, 0), i] -
                    Q_field[j,i] * dx * dy # Source/sink term, normalized by K/delta^2 implicitly in sum_coeffs
                ) / (coeff_xp + coeff_xn + coeff_yp + coeff_yn)
                
                # Manual application of no_flow for edge internal cells
                if i == nx - 1 and bc_right_type == 'no_flow':
                     h_curr[j,i] = h_curr[j,i-1] # Essentially dh/dx = 0
                if i == 0 and bc_left_type == 'no_flow':
                     h_curr[j,i] = h_curr[j,i+1]
                if j == ny - 1 and bc_top_type == 'no_flow':
                     h_curr[j,i] = h_curr[j-1,i]
                if j == 0 and bc_bottom_type == 'no_flow':
                     h_curr[j,i] = h_curr[j+1,i]


        # Calculate max change between iterations
        max_change = np.max(np.abs(h_curr - h_prev))

        # Check for convergence
        if max_change < tolerance:
            print(f"Converged after {iteration + 1} iterations. Max change: {max_change:.2e} m")
            break
        
        h_prev[:] = h_curr[:] # Update h_prev for next iteration

        if (iteration % (max_iterations // 10)) == 0 and iteration > 0:
            print(f"  Iteration {iteration}/{max_iterations}. Max change: {max_change:.2e} m")

    else:
        print(f"Reached max iterations ({max_iterations}) without convergence. Max change: {max_change:.2e} m")

    # --- Calculate Darcy Velocities from Converged Head Field ---
    vx_field = np.zeros((ny, nx))
    vy_field = np.zeros((ny, nx))

    # Calculate velocities using central differences for interior, forward/backward for boundaries
    for j in range(ny):
        for i in range(nx):
            # Vx
            if i == 0: # Forward difference at left boundary
                vx_field[j,i] = -K_field[j,i] * (h_curr[j, i+1] - h_curr[j, i]) / dx
            elif i == nx - 1: # Backward difference at right boundary
                vx_field[j,i] = -K_field[j,i] * (h_curr[j, i] - h_curr[j, i-1]) / dx
            else: # Central difference in interior
                vx_field[j,i] = -K_field[j,i] * (h_curr[j, i+1] - h_curr[j, i-1]) / (2 * dx)

            # Vy
            if j == 0: # Forward difference at bottom boundary
                vy_field[j,i] = -K_field[j,i] * (h_curr[j+1, i] - h_curr[j, i]) / dy
            elif j == ny - 1: # Backward difference at top boundary
                vy_field[j,i] = -K_field[j,i] * (h_curr[j, i] - h_curr[j-1, i]) / dy
            else: # Central difference in interior
                vy_field[j,i] = -K_field[j,i] * (h_curr[j+1, i] - h_curr[j-1, i]) / (2 * dy)
                
    print("Simulation complete. Plots will be generated.")
    return h_curr, vx_field, vy_field, x_coords, y_coords

def plot_groundwater_results(
    h_field, vx_field, vy_field, K_field, x_coords, y_coords,
    length_x, length_y, dx, dy,
    bc_left_type, bc_left_value, bc_right_type, bc_right_value,
    bc_top_type, bc_top_value, bc_bottom_type, bc_bottom_value,
    source_sink_field=None
):
    """
    Plots the hydraulic conductivity field, hydraulic head contours, and flow vectors.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Plot 1: Hydraulic Conductivity (K) Field ---
    ax0 = axes[0]
    K_norm = Normalize(vmin=np.min(K_field), vmax=np.max(K_field))
    im_K = ax0.imshow(K_field, cmap='viridis', origin='lower',
                      extent=[0, length_x, 0, length_y], norm=K_norm)
    plt.colorbar(im_K, ax=ax0, label="Hydraulic Conductivity (m/s)", shrink=0.7)
    ax0.set_title("Hydraulic Conductivity (K) Field", fontsize=14)
    ax0.set_xlabel("X-Distance (m)", fontsize=12)
    ax0.set_ylabel("Y-Distance (m)", fontsize=12)
    ax0.grid(False) # K field is smooth, no grid needed for visual clarity

    # Mark sources/sinks if present
    if source_sink_field is not None:
        source_y, source_x = np.where(source_sink_field > 0)
        sink_y, sink_x = np.where(source_sink_field < 0)
        ax0.plot(x_coords[source_x], y_coords[source_y], 'go', markersize=8, label='Injection Well')
        ax0.plot(x_coords[sink_x], y_coords[sink_y], 'rs', markersize=8, label='Pumping Well')
        ax0.legend(loc='upper right', fontsize=10)


    # --- Plot 2: Hydraulic Head Contours and Flow Vectors ---
    ax1 = axes[1]
    head_norm = Normalize(vmin=np.min(h_field), vmax=np.max(h_field))
    im_h = ax1.imshow(h_field, cmap='RdBu_r', origin='lower',
                      extent=[0, length_x, 0, length_y], norm=head_norm)
    plt.colorbar(im_h, ax=ax1, label="Hydraulic Head (m)", shrink=0.7)
    
    # Add head contours
    # Adjust contour levels based on min/max head for clear visualization
    num_contours = 15
    levels = np.linspace(np.min(h_field), np.max(h_field), num_contours)
    contour = ax1.contour(x_coords, y_coords, h_field, levels=levels, colors='k', linestyles='--', linewidths=0.8)
    ax1.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Add flow vectors (quiver plot)
    # Subsample vectors to avoid cluttering the plot
    skip = max(1, int(max(nx, ny) / 15)) # Adjust skip based on grid size
    ax1.quiver(x_coords[::skip], y_coords[::skip], vx_field[::skip,::skip], vy_field[::skip,::skip],
               color='w', scale=np.max(np.sqrt(vx_field**2 + vy_field**2)) * 10,  # Scale vectors
               width=0.003, headwidth=5, headlength=7, alpha=0.8)

    ax1.set_title("Hydraulic Head & Flow Vectors", fontsize=14)
    ax1.set_xlabel("X-Distance (m)", fontsize=12)
    ax1.set_ylabel("Y-Distance (m)", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Add text labels for boundary conditions
    # Top/Bottom
    ax1.text(length_x/2, length_y * 1.02, f"Top BC: {bc_top_type.replace('_',' ')}", 
             ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(length_x/2, length_y * -0.02, f"Bottom BC: {bc_bottom_type.replace('_',' ')}", 
             ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    # Left/Right
    ax1.text(-0.02 * length_x, length_y/2, f"Left BC: {bc_left_type.replace('_',' ')}", 
             ha='right', va='center', rotation=90, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(1.02 * length_x, length_y/2, f"Right BC: {bc_right_type.replace('_',' ')}", 
             ha='left', va='center', rotation=-90, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))


    plt.tight_layout()
    plt.show()

def create_layered_k_field(nx, ny, dx, dy, x_coords, y_coords, layer_definitions):
    """
    Creates a 2D hydraulic conductivity field based on layer definitions.
    Layer definitions: list of {'type': 'horizontal'/'vertical', 'position': float, 'k_value': float}
    or {'shape': 'rectangle', 'x_range': (x1,x2), 'y_range': (y1,y2), 'k_value': float}
    Base K is applied first, then higher layers/shapes overwrite.
    """
    K_field = np.full((ny, nx), layer_definitions['base_k'])

    for layer in layer_definitions['layers']:
        if layer['type'] == 'horizontal':
            y_interface_idx = int(round(layer['position'] / dy))
            # Apply K to cells below/above interface
            if layer.get('apply_below', True): # Default to apply below interface
                K_field[:y_interface_idx, :] = layer['k_value']
            else: # Apply above interface
                K_field[y_interface_idx:, :] = layer['k_value']
        
        elif layer['type'] == 'vertical':
            x_interface_idx = int(round(layer['position'] / dx))
            if layer.get('apply_left', True):
                K_field[:, :x_interface_idx] = layer['k_value']
            else:
                K_field[:, x_interface_idx:] = layer['k_value']
        
        elif layer['type'] == 'rectangle':
            x1_idx = int(round(layer['x_range'][0] / dx))
            x2_idx = int(round(layer['x_range'][1] / dx))
            y1_idx = int(round(layer['y_range'][0] / dy))
            y2_idx = int(round(layer['y_range'][1] / dy))
            K_field[y1_idx:y2_idx+1, x1_idx:x2_idx+1] = layer['k_value']
        
        elif layer['type'] == 'circle': # Example: low K inclusion
            center_x, center_y = layer['center']
            radius = layer['radius']
            for j in range(ny):
                for i in range(nx):
                    if (x_coords[i] - center_x)**2 + (y_coords[j] - center_y)**2 <= radius**2:
                        K_field[j,i] = layer['k_value']
                        
    return K_field

def create_source_sink_field(nx, ny, dx, dy, x_coords, y_coords, well_definitions):
    """
    Creates a 2D source/sink field based on well definitions.
    Well definitions: list of {'x': x_coord, 'y': y_coord, 'flow_rate': float (m^3/s)}
    Positive flow_rate for injection, negative for pumping.
    """
    Q_field = np.zeros((ny, nx))
    for well in well_definitions:
        x_idx = int(round(well['x'] / dx))
        y_idx = int(round(well['y'] / dy))
        if 0 <= x_idx < nx and 0 <= y_idx < ny:
            Q_field[y_idx, x_idx] = well['flow_rate'] / (dx * dy) # Convert total flow rate to Q per unit area
        else:
            print(f"Warning: Well at ({well['x']},{well['y']}) is outside grid. Ignoring.")
    return Q_field


if __name__ == "__main__":
    # --- Define Simulation Parameters ---
    x_domain_length = 200.0 # meters
    y_domain_length = 150.0 # meters
    dx = 2.0                # Spatial grid spacing (m)
    dy = 2.0                # Spatial grid spacing (m)

    # --- Iterative Solver Parameters ---
    solver_tolerance = 1e-5 # Convergence tolerance for head (m)
    max_solver_iterations = 10000 # Max iterations for the solver

    # --- Hydraulic Conductivity (K) Field Definition ---
    # Define layers or heterogeneities within the aquifer
    # Base K is the default value for the entire domain.
    # Layers/shapes defined in 'layers' list will overwrite.
    K_field_definitions = {
        'base_k': 1e-4, # m/s (e.g., sandy loam)
        'layers': [
            # Example 1: Low K layer (clay lens) in the middle
            {'type': 'rectangle', 'x_range': (70, 130), 'y_range': (40, 80), 'k_value': 1e-6}, # Clay lens
            # Example 2: High K channel/fault
            {'type': 'rectangle', 'x_range': (160, 170), 'y_range': (0, 150), 'k_value': 5e-3}, # High K channel (fault zone)
            # Example 3: Different base material in upper part
            {'type': 'horizontal', 'position': 100, 'k_value': 5e-5, 'apply_below': False} # Silt layer above 100m
        ]
    }

    # --- Source/Sink (Wells) Definition ---
    # Define pumping (negative flow) or injection (positive flow) wells
    well_definitions = [
        # {'x': 25.0, 'y': 75.0, 'flow_rate': -0.005}, # Pumping well (m^3/s)
        # {'x': 180.0, 'y': 25.0, 'flow_rate': 0.002}  # Injection well (m^3/s)
    ]

    # --- Boundary Conditions ---
    # Options: 'fixed_head' (Dirichlet), 'no_flow' (Neumann)
    
    # Scenario: Flow from left to right, with no flow at top/bottom
    bc_left_type_val = 'fixed_head'; bc_left_value_val = 10.0 # m
    bc_right_type_val = 'fixed_head'; bc_right_value_val = 5.0 # m
    bc_top_type_val = 'no_flow'; bc_top_value_val = None
    bc_bottom_type_val = 'no_flow'; bc_bottom_value_val = None

    # --- Initial Head Field (Optional) ---
    # A simple linear gradient as an initial guess can speed up convergence
    initial_head_guess = np.linspace(bc_left_value_val, bc_right_value_val, int(x_domain_length / dx) + 1)
    initial_head_field_val = np.tile(initial_head_guess, (int(y_domain_length / dy) + 1, 1))

    # --- Generate K and Q fields ---
    nx_grid = int(x_domain_length / dx) + 1
    ny_grid = int(y_domain_length / dy) + 1
    x_coords_grid = np.linspace(0, x_domain_length, nx_grid)
    y_coords_grid = np.linspace(0, y_domain_length, ny_grid)

    K_field_generated = create_layered_k_field(nx_grid, ny_grid, dx, dy, x_coords_grid, y_coords_grid, K_field_definitions)
    Q_field_generated = create_source_sink_field(nx_grid, ny_grid, dx, dy, x_coords_grid, y_coords_grid, well_definitions)

    # --- Run Simulation ---
    h_field_result, vx_field_result, vy_field_result, x_coords_result, y_coords_result = simulate_2d_groundwater_flow(
        length_x=x_domain_length,
        length_y=y_domain_length,
        dx=dx,
        dy=dy,
        hydraulic_conductivity_field=K_field_generated,
        source_sink_field=Q_field_generated,
        bc_left_type=bc_left_type_val, bc_left_value=bc_left_value_val,
        bc_right_type=bc_right_type_val, bc_right_value=bc_right_value_val,
        bc_top_type=bc_top_type_val, bc_top_value=bc_top_value_val,
        bc_bottom_type=bc_bottom_type_val, bc_bottom_value=bc_bottom_value_val,
        tolerance=solver_tolerance,
        max_iterations=max_solver_iterations,
        initial_head_field=initial_head_field_val
    )

    # --- Plot Results ---
    plot_groundwater_results(
        h_field_result, vx_field_result, vy_field_result, K_field_generated,
        x_coords_result, y_coords_result,
        x_domain_length, y_domain_length, dx, dy,
        bc_left_type_val, bc_left_value_val, bc_right_type_val, bc_right_value_val,
        bc_top_type_val, bc_top_value_val, bc_bottom_type_val, bc_bottom_value_val,
        source_sink_field=Q_field_generated
    )

    print("\n2D Steady-State Groundwater Flow simulation complete. Check the generated plots.")
