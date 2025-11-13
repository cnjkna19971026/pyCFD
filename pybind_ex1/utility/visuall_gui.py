import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Import our custom C++ module ---
try:
    import cpp_solver_module
except ImportError:
    messagebox.showerror("Import Error", "Could not find 'cpp_solver_module'.\nPlease ensure it's compiled and in the same directory.")
    exit()

class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("C++ Solver Interface")

        # --- Create Main Frames ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        ttk.Label(control_frame, text="Num Steps:").grid(row=0, column=0, sticky="w")
        self.num_steps = tk.IntVar(value=200)
        ttk.Entry(control_frame, textvariable=self.num_steps).grid(row=0, column=1)

        ttk.Label(control_frame, text="Radius:").grid(row=1, column=0, sticky="w")
        self.radius = tk.DoubleVar(value=10.0)
        ttk.Entry(control_frame, textvariable=self.radius).grid(row=1, column=1)

        ttk.Label(control_frame, text="Ang. Velocity:").grid(row=2, column=0, sticky="w")
        self.angular_velocity = tk.DoubleVar(value=0.1)
        ttk.Entry(control_frame, textvariable=self.angular_velocity).grid(row=2, column=1)
        
        # --- Buttons ---
        run_button = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=3, column=0, columnspan=2, pady=10)

        quit_button = ttk.Button(control_frame, text="Quit", command=root.destroy)
        quit_button.grid(row=4, column=0, columnspan=2)

        # --- Matplotlib Plot Setup ---
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add Matplotlib's navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_title("C++ Particle Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True)
        self.ax.axis('equal')
        self.canvas.draw()

    def run_simulation(self):
        """Callback for the 'Run Simulation' button."""
        try:
            # 1. Get parameters from the GUI
            steps = self.num_steps.get()
            rad = self.radius.get()
            vel = self.angular_velocity.get()
            
            # 2. Call the C++ function directly
            print(f"Calling C++ solver with: steps={steps}, radius={rad}, velocity={vel}")
            trajectory = cpp_solver_module.run_simulation(
                num_steps=steps,
                radius=rad,
                angular_velocity=vel
            )
            print("C++ solver finished.")
            
            # 3. Update the plot with the results
            self.update_plot(trajectory)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all parameters.")
        except Exception as e:
            messagebox.showerror("Runtime Error", f"An error occurred: {e}")

    def update_plot(self, trajectory):
        """Clears the plot and draws the new trajectory."""
        self.init_plot() # Clear and reset axes
        
        if not trajectory:
            return

        # Extract x and y coordinates from our list of Point objects
        x_coords = [p.x for p in trajectory]
        y_coords = [p.y for p in trajectory]
        
        self.ax.plot(x_coords, y_coords, 'o-', label=f'Trajectory ({len(trajectory)} steps)')
        self.ax.legend()
        
        # Redraw the canvas to show the new plot
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()
