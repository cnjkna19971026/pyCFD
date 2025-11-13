import pyvista as pv
import numpy as np

# Set a profesional plotting theme
pv.set_plot_theme("document")

# 1. READ THE VTK FILE
# PyVista automatically detects the file type and loads the data.
filename = input('Enter .vtk file ')
try:
    grid = pv.read(filename)
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please run the C++ program first to generate it.")
    exit()

# 2. INSPECT THE LOADED DATA
# This is a great practice to see what you've loaded.
# It will show dimensions, number of points, and the available data arrays.
print("--- Loaded VTK Data ---")
print(grid)
print("\nAvailable data arrays:", grid.point_data.keys())
print("-" * 25)


# 3. VISUALIZE THE DATA
# We will create a plotter with two subplots side-by-side.
plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 600])

# --- Subplot 1: Scalar Field (Temperature) ---
plotter.subplot(0, 0)
plotter.add_text("Scalar Field (Temperature)", font_size=15)
# Add the grid mesh, colored by the 'Temperature' scalar data.
plotter.add_mesh(grid, scalars='Temperature', cmap='hot', show_edges=True)
# Set the camera to a 2D view looking down the Z-axis.
plotter.view_xy()


# --- Subplot 2: Vector Field (Gradient) ---
plotter.subplot(0, 1)
plotter.add_text("Gradient Vector Field", font_size=15)
# First, create glyphs (arrows) from the vector data.
# The 'factor' scales the arrow length for better visibility.
arrows = grid.glyph(orient='Gradient', scale='Gradient', factor=0.01)

# Add the glyphs (arrows) to the plot.
plotter.add_mesh(arrows, color='blue')
# Add the wireframe of the grid in the background for context.
plotter.add_mesh(grid, style='wireframe', color='grey', opacity=0.5)
plotter.view_xy()

# Link the cameras so that when you zoom/pan one, the other follows.
plotter.link_views()

# Display the plot.
print("Showing plot. Close the window to exit.")
plotter.show()
