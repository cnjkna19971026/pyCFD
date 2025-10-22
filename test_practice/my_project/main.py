import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定義 CNN 模型 (Encoder-Decoder 架構) ---


# Use input_shape to define your model (Encoder-decoder Architeture)
#  

def build_3d_cfd_cnn_model(input_shape=(64 , 64 , 64 , 3)):
    """
    建立一個簡化的 Encoder-Decoder CNN 模型，用於CFD問題。

    Args:
        input_shape (tuple): 輸入數據的形狀 (高度, 寬度, 通道數)。
                             例如：(64, 64, 2) 表示 64x64 網格，2個速度分量。

    Returns:
        tf.keras.Model: 編譯好的 CNN 模型。
    """
    inputs = layers.Input(shape=input_shape)

    # --- 編碼器 (Encoder) ---
    # 卷積層提取特徵，池化層縮小空間尺寸
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs) # 32個濾波器
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x) # 尺寸變為 32x32

    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x) # 64個濾波器
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x) # 尺寸變為 16x16

    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x) # 128個濾波器
    encoder_output = layers.MaxPooling3D((2, 2, 2), padding='same')(x) # 尺寸變為 8x8
    # 此時 encoder_output 的形狀為 (None, 8, 8, 128)

    # --- 解碼器 (Decoder) ---
    # 反卷積層 (Conv2DTranspose) 恢復空間尺寸，增加細節
    x = layers.Conv3DTranspose(128, (3, 3, 3), activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling3D((2, 2, 2))(x) # 尺寸變為 16x16

    x = layers.Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x) # 尺寸變為 32x32

    x = layers.Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x) # 尺寸變為 64x64

    # 輸出層：1個通道，表示壓力場
    # 'linear' 激活函數用於迴歸問題，因為壓力值可以是任意連續數值
    outputs = layers.Conv3D(1, (3, 3, 3), activation='linear', padding='same')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 編譯模型：
    # optimizer='adam' 是一個常用的優化器
    # loss='mse' (均方誤差) 是迴歸問題的標準損失函數
    model.compile(optimizer='adam', loss='mse')

    return model

# --- 2. 生成模擬數據 (替代真實 CFD 數據) ---
# 在實際應用中，這裡會載入你使用 OpenFOAM, Fluent 等軟體生成的 CFD 數據集

def generate_mock_cfd_data(num_samples=100, grid_size=64):
    """
    生成模擬的 CFD 數據集。
    輸入：隨機的速度場 (u, v)
    輸出：模擬的壓力場 (p)，這裡我們做一個簡單的函數關係，
          例如 P = f(u^2 + v^2)
    """
    X = np.zeros((num_samples, grid_size, grid_size, grid_size, 3)) # 輸入：u, v ,w速度場
    Y = np.zeros((num_samples, grid_size, grid_size, grid_size, 1)) # 輸出：壓力 p

    for i in range(num_samples):
        # 模擬隨機的速度場
        # 這裡生成的是平滑的噪聲，稍微模擬真實流場的連續性
        u = np.random.rand(grid_size, grid_size, grid_size) * 2 - 1 # -1 到 1
        v = np.random.rand(grid_size, grid_size, grid_size) * 2 - 1 # -1 到 1
        w = np.random.rand(grid_size, grid_size, grid_size) * 2 - 1 # -1 到 1

        # 使用高斯模糊使其更平滑一點
        from scipy.ndimage import gaussian_filter
        u = gaussian_filter(u, sigma=2)
        v = gaussian_filter(v, sigma=2)
        w = gaussian_filter(w, sigma=2)

        X[i, :, :, :, 0] = u
        X[i, :, :, :, 1] = v
        X[i, :, :, :, 2] = w

        # 簡單地模擬壓力場作為速度平方和的函數，加上一些噪聲
        # 這是為了讓模型有東西可以學習，實際 CFD 關係更複雜
        pressure = -(u**2 + v**2 + w**2) + np.random.rand(grid_size, grid_size) * 0.1 # 模擬壓力 P = -rho/2 * (u^2 + v^2)
        Y[i, :, :, :, 0] = pressure

    # 將數據正規化 (歸一化) 到 0-1 範圍或 -1 到 1 範圍，有助於訓練穩定性
    # 這裡簡單地除以最大絕對值，實際中通常使用 Min-Max Scaler 或 StandardScaler
    X_max_abs = np.max(np.abs(X))
    Y_max_abs = np.max(np.abs(Y))
    if X_max_abs > 0:
        X /= X_max_abs
    if Y_max_abs > 0:
        Y /= Y_max_abs

    return X, Y

#
# post-process _ data tranformation
#

import pyvista as pv
import numpy as np

def save_to_vtk(filename, velocity_field , pressure_field):
    """
    saves a 4-channel 3d pytorch tensor to a vtk file for paraview.

    args:
        filename (str): the path to save the .vtk file.
        data_tensor (torch.tensor): the model output tensor with shape
                                    (1, 4, d, h, w), where channels are p, u, v, w.
    """

    # 2. remove the batch dimension (shape becomes [4, d, h, w])
    velo_np = np.squeeze(velocity_field, axis=0)
    pres_np = np.squeeze(pressure_field, axis=0)
    
    # 3. get dimensions (depth, height, width)
    d, h, w, _  = velo_np.shape
    if velo_np.shape[:3] != pres_np.shape[:3]:
        raise ValueError(f"Velocity dimensions {velo_np.shape[:3]} do not match pressure dimensions {pres_np.shape[:3]}")

    # 4. create a pyvista uniformgrid object
    # dimensions are specified as (nx, ny, nz) which corresponds to (w, h, d)
    grid = pv.ImageData()
    grid.dimensions = (w, h, d)
    # optional: if your grid isn't from (0,0,0) or has different spacing
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)

    
 # add scalar data. the default flatten() is order='c', which is correct for vtk.
    # it arranges the data with x varying fastest, then y, then z.
    grid.point_data['pressure'] = np.squeeze(pres_np,axis=-1).flatten(order='C')
    
    # combine u, v, w into a single vector array for vtk
    # we must stack in (u, v, w) order and reshape.
    grid.point_data['velocity'] = velo_np.reshape(-1,3)

    # verify that the number of points matches the data size
    if grid.n_points != pres_np.size:
        raise ValueError(
            f"mismatch in point count ({grid.n_points}) and data size ({pressure.size})."
            )

    # 6. save the grid to a file
    grid.save(filename)
    print(f"successfully saved prediction to {filename}")

# --- post-process

# def visulization():




# --- 3. 執行模型訓練和預測 ---
if __name__ == "__main__":
    # --- A. 準備數據 ---
    NUM_SAMPLES = 50 # 生成的數據樣本數
    GRID_SIZE = 32    # 網格大小
    X_data, Y_data = generate_mock_cfd_data(NUM_SAMPLES, GRID_SIZE)

    # 劃分訓練集和測試集
    split_ratio = 0.8
    split_index = int(NUM_SAMPLES * split_ratio)

    X_train, Y_train = X_data[:split_index], Y_data[:split_index]
    X_test, Y_test = X_data[split_index:], Y_data[split_index:]

    print(f"訓練集形狀 (X_train): {X_train.shape}, (Y_train): {Y_train.shape}")
    print(f"測試集形狀 (X_test): {X_test.shape}, (Y_test): {Y_test.shape}")
    print(f"NUM_SAMPLES : ", NUM_SAMPLES,f"GRID_SIZE : " ,GRID_SIZE)
    

    # --- b. 建立和訓練模型 ---
    model = build_3d_cfd_cnn_model(input_shape=(GRID_SIZE, GRID_SIZE ,  GRID_SIZE ,3))
    model.summary() # 打印模型結構概覽

    print("\n--- 開始訓練模型 ---")
    history = model.fit(X_train, Y_train,
                        epochs=100,          # 訓練輪次
                        batch_size=10,       # 每次更新模型的樣本數
                        validation_split=0.1, # 從訓練集中分出10%作為驗證集
                        verbose=1)

    print("\n--- 評估模型 ---")
    loss = model.evaluate(X_test, Y_test, verbose=0)
    print(f"測試集 mse 損失: {loss:.4f}")

    # --- c. 進行預測並可視化結果 ---
    print("\n--- 進行預測並可視化 ---")
    # 選擇一個測試樣本進行預測
    sample_index = np.random.randint(0, len(X_test)) # 隨機選擇一個測試樣本
    input_velocity_field = X_test[sample_index:sample_index+1] # 注意切片使其保持批次維度
    true_pressure_field = Y_test[sample_index:sample_index+1]

    predicted_pressure_field = model.predict(input_velocity_field)

    #### save  predict data as .vtk
    save_to_vtk("test.vtk"  ,input_velocity_field, predicted_pressure_field)
    
    #### save ground data as .vtk
    save_to_vtk("ground.vtk",input_velocity_field, true_pressure_field)


    # 可視化結果
    
    #### turn 3D data into 2D
    slice_idx = GRID_SIZE // 2


    plt.figure(figsize=(18, 6))

    # 1. 輸入速度場 (x_test 的第0個通道，即 u 速度)
    plt.subplot(1, 5, 1)
    plt.imshow(input_velocity_field[0, slice_idx, :, :, 0], cmap='viridis') # 顯示 u 速度分量
    plt.title('input u velocity field')
    plt.colorbar(label='normalized u')
    plt.axis('off')

    # 2. 輸入速度場 (x_test 的第1個通道，即 v 速度)
    plt.subplot(1, 5, 2)
    plt.imshow(input_velocity_field[0, slice_idx, :, :, 1], cmap='viridis') # 顯示 v 速度分量
    plt.title('input v velocity field')
    plt.colorbar(label='normalized v')
    plt.axis('off')

    # 3. 輸入速度場 (x_test 的第2個通道，即 w 速度)
    plt.subplot(1, 5, 3)
    plt.imshow(input_velocity_field[0, slice_idx, :, :, 2], cmap='viridis') # 顯示 v 速度分量
    plt.title('input w velocity field')
    plt.colorbar(label='normalized w')
    plt.axis('off')

    # 3. 實際的壓力場
    plt.subplot(1, 5, 4)
    plt.imshow(true_pressure_field[0, slice_idx, :, :, 0], cmap='plasma') # 顯示真實壓力場
    plt.title('true pressure field')
    plt.colorbar(label='normalized pressure')
    plt.axis('off')

    # 4. 預測的壓力場
    plt.subplot(1, 5, 5)
    plt.imshow(predicted_pressure_field[0, slice_idx, :, :, 0], cmap='plasma') # 顯示預測壓力場
    plt.title('predicted pressure field')
    plt.colorbar(label='normalized pressure')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 繪製訓練損失
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('model loss over epochs')
    plt.xlabel('epoch')
    plt.ylabel('mean squared error (mse)')
    plt.legend()
    plt.grid(True)
    plt.show()

