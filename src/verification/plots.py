import numpy as np
import plotly.graph_objects as go

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