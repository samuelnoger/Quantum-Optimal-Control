import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_chsh_results(
    correlators,
    chsh_values,
    chsh_errors,
    *,
    shots=5000,
    figsize=(11, 4),
):
    """Plot CHSH correlators and total S values for one experiment."""
    labels = list(correlators)
    values = list(correlators.values())
    ideal_values = [1 / np.sqrt(2)] * 3 + [-1 / np.sqrt(2)]
    x = np.arange(len(labels))

    fig, (ax_correlators, ax_chsh) = plt.subplots(1, 2, figsize=figsize)
    ax_correlators.bar(
        x - 0.18,
        ideal_values,
        width=0.35,
        alpha=0.5,
        color="gray",
        label=r"Ideal $\Phi^+$ ($\pm 1/\sqrt{2}$)",
    )
    ax_correlators.bar(
        x + 0.18,
        values,
        width=0.35,
        color="teal",
        label=f"QuTiP Fridge ({shots:,} shots)",
    )
    ax_correlators.axhline(0, color="black", linewidth=0.8)
    ax_correlators.set(
        xticks=x,
        xticklabels=labels,
        ylim=(-1.0, 1.0),
        ylabel="Parity Correlator",
        title="CHSH Parity Correlators on Hardware",
    )
    ax_correlators.legend()
    ax_correlators.grid(True, alpha=0.3)

    categories = ["Uncorrected\nFrame", "Fridge +\nVirtual-Z", r"Ideal $\Phi^+$"]
    colors = ["purple", "teal", "royalblue"]
    ax_chsh.bar(
        categories,
        chsh_values,
        yerr=chsh_errors,
        capsize=6,
        color=colors,
        alpha=0.85,
        width=0.4,
    )
    ax_chsh.axhline(
        2.0,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label="Classical Bound (S = 2.0)",
    )
    ax_chsh.axhline(
        2 * np.sqrt(2),
        color="navy",
        linestyle=":",
        linewidth=2,
        label=r"Quantum Bound ($2\sqrt{2} \approx 2.828$)",
    )
    ax_chsh.set(
        ylim=(0, 3.2),
        ylabel="CHSH Parameter S",
        title="Bell Inequality Violation",
    )
    ax_chsh.legend(loc="lower right")
    ax_chsh.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def plot_absolute_cityscape(rho_matrix, title="Analog Bell State: True Physical Entanglement"):
    """
    Renders a 3D Cityscape where height = absolute magnitude, 
    and color = complex phase angle in radians.
    """
    rho_abs = np.abs(rho_matrix)
    rho_angle = np.angle(rho_matrix)
    
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    traces = []
    
    # Define the 12 triangles composing a single 3D rectangular pillar
    # Vertices 0-3 are the bottom square, 4-7 are the top square
    i_faces = [0, 0, 4, 4, 0, 0, 2, 2, 1, 1, 0, 0]
    j_faces = [1, 2, 5, 6, 1, 5, 3, 7, 2, 6, 3, 7]
    k_faces = [2, 3, 6, 7, 5, 4, 7, 6, 6, 5, 7, 4]
    
    w = 0.35  # Half-width of the pillars to leave spacing between them
    
    for r in range(4):
        for c in range(4):
            z_val = rho_abs[r, c]
            p_val = rho_angle[r, c]
            
            # Generate the 8 spatial vertices for this specific pillar
            x_corners = [c-w, c+w, c+w, c-w, c-w, c+w, c+w, c-w]
            y_corners = [r-w, r-w, r+w, r+w, r-w, r-w, r+w, r+w]
            z_corners = [0, 0, 0, 0, z_val, z_val, z_val, z_val]
            
            # Create a rich interactive hover tooltip
            hover_text = [
                f"Row: {labels[r]}<br>Col: {labels[c]}<br>Magnitude: {z_val:.4f}<br>Phase: {p_val:.4f} rad"
            ] * 8
            
            # Only draw the colorbar for the very first trace to avoid duplicates
            show_scale = True if (r == 0 and c == 0) else False
            
            trace = go.Mesh3d(
                x=x_corners, 
                y=y_corners, 
                z=z_corners,
                i=i_faces, 
                j=j_faces, 
                k=k_faces,
                intensity=[p_val]*8, # Map the color to the phase angle
                intensitymode="vertex",
                colorscale='Twilight',        # Cyclic colormap for phase (-pi to pi)
                cmin=-np.pi, 
                cmax=np.pi,
                showscale=show_scale,
                colorbar = dict(
                title="Phase (rad)", 
                    tickvals=[-np.pi, 0, np.pi], 
                    ticktext=["−π", "0", "π"]
                ),
                hoverinfo="text",
                text=hover_text,
                flatshading=True,          
            )
            traces.append(trace)
            
    fig = go.Figure(data=traces)
    
    # Format the grid, axes, and viewing angle
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(tickvals=[0, 1, 2, 3], ticktext=labels, title=""),
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=labels, title=""),
            zaxis=dict(title="Absolute Magnitude", range=[0, 0.55]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6), # Slightly squashes the Z axis for a better aesthetic
        ),
        width=600, 
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig