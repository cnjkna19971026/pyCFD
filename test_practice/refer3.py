import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the CNN Architecture (Same as before)
class SimpleCFD_CNN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(SimpleCFD_CNN3D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # -> (N, 16, 32, 16, 16)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # -> (N, 32, 16, 8, 8)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2), # -> (N, 16, 32, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2), # -> (N, 8, 64, 32, 32)
            nn.ReLU(inplace=True),
        )
        # Final Output Layer
        self.final_conv = nn.Conv3d(8, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x

# 2. Prepare for Training (Using structured Dummy Data)

# --- Parameters ---
batch_size = 4
in_channels = 1   # Geometry (obstacle vs. fluid)
out_channels = 4  # p, u, v, w
D, H, W = 64, 32, 32 # Depth, Height, Width

# --- Create *Structured* Dummy Data ---
# This makes visualization more intuitive than random noise.
# We will create a dataset with a solid block in the middle.
def create_structured_data(batch_size, D, H, W):
    # Input: A 3D volume representing the geometry (0 for fluid, 1 for solid)
    input_geom = torch.zeros(batch_size, 1, D, H, W)
    # Target: A 4-channel 3D volume representing the flow field
    target_flow = torch.zeros(batch_size, 4, D, H, W)

    for i in range(batch_size):
        # Create a solid block obstacle in the middle
        d_start, d_end = D//2 - 5, D//2 + 5
        h_start, h_end = H//2 - 5, H//2 + 5
        w_start, w_end = W//2 - 5, W//2 + 5
        input_geom[i, 0, d_start:d_end, h_start:h_end, w_start:w_end] = 1.0

        # Create a simple fake flow field around the block (e.g., U-velocity)
        # Velocity is 1.0 upstream, and 0 in the "shadow" of the block
        u_velocity = torch.linspace(1, 0.2, D).view(1, D, 1, 1).repeat(1, 1, H, W)
        u_velocity[:, d_start:, :, :] = 0 # Simple shadow
        target_flow[i, 1, :, :, :] = u_velocity # Assign to U channel

    return input_geom, target_flow

dummy_input, dummy_target = create_structured_data(batch_size, D, H, W)

# --- Instantiate Model, Loss Function, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleCFD_CNN3D(in_channels=in_channels, out_channels=out_channels).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. The Training Loop (Same as before)
num_epochs = 20
print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    inputs = dummy_input.to(device)
    targets = dummy_target.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
print("--- Training Finished ---")


# 4. Inference and Visualization
print("\n--- Running Inference and Visualization ---")

def visualize_slice(input_tensor, output_tensor, slice_idx, filename=None):
    """
    Visualizes a 2D slice from the center of the 3D input and output tensors.
    
    Args:
        input_tensor (torch.Tensor): The input geometry tensor (N, 1, D, H, W).
        output_tensor (torch.Tensor): The predicted flow field (N, 4, D, H, W).
        slice_idx (int): The index of the slice to visualize along the D (depth) axis.
    """
    # Ensure tensors are on the CPU and converted to NumPy for plotting
    input_np = input_tensor.cpu().detach().numpy()
    output_np = output_tensor.cpu().detach().numpy()

    # We only visualize the first item in the batch
    input_slice = input_np[0, 0, slice_idx, :, :]
    output_slices = output_np[0, :, slice_idx, :, :]
    
    field_names = ['Pressure (p)', 'U-velocity', 'V-velocity', 'W-velocity']
    
    # Create a 1x5 subplot (1 for input, 4 for output fields)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot Input Geometry
    ax = axes[0]
    im = ax.imshow(input_slice, cmap='binary', origin='lower')
    ax.set_title('Input Geometry')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot Output Fields
    for i in range(4):
        ax = axes[i+1]
        # Use a diverging colormap for velocities, which can be +/-
        cmap = 'viridis' if i == 0 else 'coolwarm'
        im = ax.imshow(output_slices[i, :, :], cmap=cmap, origin='lower')
        ax.set_title(f'Predicted {field_names[i]}')
        ax.set_xlabel('Width')
        # ax.set_yticklabels([]) # Hide y-axis labels for cleaner look
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Center Slice Visualization (Depth Index: {slice_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        
    plt.show()

import pyvista as pv
import numpy as np

def save_to_vtk(filename, data_tensor):
    """
    Saves a 4-channel 3D PyTorch tensor to a VTK file for ParaView.

    Args:
        filename (str): The path to save the .vtk file.
        data_tensor (torch.Tensor): The model output tensor with shape
                                    (1, 4, D, H, W), where channels are p, u, v, w.
    """
    # 1. Ensure tensor is on CPU and convert to NumPy
    if data_tensor.is_cuda:
        data_tensor = data_tensor.cpu()
    data_np = data_tensor.detach().numpy()

    # 2. Remove the batch dimension (shape becomes [4, D, H, W])
    data_np = np.squeeze(data_np, axis=0)
    
    # 3. Get dimensions (Depth, Height, Width)
    _, D, H, W = data_np.shape

    # 4. Create a PyVista UniformGrid object
    # Dimensions are specified as (nx, ny, nz) which corresponds to (W, H, D)
    grid = pv.ImageData()
    grid.dimensions = (W, H, D)
    # Optional: If your grid isn't from (0,0,0) or has different spacing
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)

    # 5. Separate the channels and add to the grid as point data
    pressure = data_np[0] # Shape: (D, H, W)
    u_vel = data_np[1]
    v_vel = data_np[2]
    w_vel = data_np[3]
    
 # Add scalar data. The default flatten() is order='C', which is correct for VTK.
    # It arranges the data with X varying fastest, then Y, then Z.
    grid.point_data['pressure'] = pressure.flatten()
    
    # Combine u, v, w into a single vector array for VTK
    # We must stack in (u, v, w) order and reshape.
    velocity = np.vstack((u_vel.flatten(), v_vel.flatten(), w_vel.flatten())).transpose()
    grid.point_data['velocity'] = velocity

    # Verify that the number of points matches the data size
    if grid.n_points != pressure.size:
        raise ValueError(
            f"Mismatch in point count ({grid.n_points}) and data size ({pressure.size})."
            )

    # 6. Save the grid to a file
    grid.save(filename)
    print(f"Successfully saved prediction to {filename}")



# ... (All the previous code for training and setup remains the same) ...


if __name__ == "__main__":
    # 4. Inference and Visualization
    print("\n--- Running Inference and Visualization ---")
    
    model.eval()
    with torch.no_grad():
        sample_input = dummy_input[0:1].to(device) # Shape: (1, 1, 64, 32, 32)
        predicted_flow_field = model(sample_input)
    
    # --- DEBUGGING: Check the output tensor's statistics ---
    p_pred = predicted_flow_field[0, 0, ...]
    u_pred = predicted_flow_field[0, 1, ...]
    print("\n--- Predicted Field Statistics ---")
    print(f"Pressure Stats: Min={p_pred.min():.4f}, Max={p_pred.max():.4f}, Mean={p_pred.mean():.4f}")
    print(f"U-Velocity Stats: Min={u_pred.min():.4f}, Max={u_pred.max():.4f}, Mean={u_pred.mean():.4f}")
    print("------------------------------------")
    # If these values are all very close to zero, the VTK file will appear empty.
    
    # --- NEW PART: EXPORT TO VTK ---
    save_to_vtk("prediction.vtk", predicted_flow_field)
    # --- END NEW PART ---
    
    # Visualize a slice from the middle of the domain's depth using Matplotlib
    slice_to_show = D // 2
    visualize_slice(sample_input, predicted_flow_field, slice_idx=slice_to_show, filename="cfd_cnn_prediction.png")

