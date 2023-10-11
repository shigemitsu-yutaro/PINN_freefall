import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. PINN class definition
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        # Change architecture to have 4 layers with 20 nodes each
        self.dense_1 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.dense_2 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.dense_3 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.dense_4 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.output_layer(x)
        return x
    
# 2. FreeFallPINN class definition
class FreeFallPINN(PINN):
    def __init__(self, gravity=9.81):
        super(FreeFallPINN, self).__init__()
        self.gravity = tf.constant(gravity, dtype=tf.float32)

    def compute_physical_loss(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred = self(x)
            dy_dt = tape.gradient(y_pred, x)
        d2y_dt2 = tape.gradient(dy_dt, x)
        
        # Physical loss: Difference between the computed acceleration and gravity
        gravity_loss = tf.reduce_mean(tf.square(d2y_dt2 + self.gravity))
        
        # Initial conditions loss
        initial_conditions_loss = tf.square(self(tf.constant([[0.]], dtype=tf.float32))) + \
                                  tf.square(dy_dt[0]) + \
                                  tf.square(d2y_dt2[0])
        
        # New condition: The velocity should be equal to gravity multiplied by time
        velocity_condition_loss = tf.reduce_mean(tf.square(dy_dt + self.gravity * x))
        
        return gravity_loss, velocity_condition_loss, initial_conditions_loss

    # For MSE training
    def mse_train_step(self, data):
        x = data[:, 0:1]
        y = data[:, 1:2]
        with tf.GradientTape() as tape:
            y_pred = self(x)
            mse_loss = tf.keras.losses.MSE(y, y_pred)
        grads = tape.gradient(mse_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'mse_loss': mse_loss, 'physical_loss': tf.constant(0.0, dtype=tf.float32)}

    # For physics-based training
    def physics_train_step(self, x):
        x = tf.expand_dims(x, axis=-1)  # Add this line
        with tf.GradientTape() as tape:
            gravity_loss, velocity_condition_loss, initial_conditions_loss = self.compute_physical_loss(x)
            physical_loss = 100*gravity_loss + 100*velocity_condition_loss + initial_conditions_loss
        grads = tape.gradient(physical_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'mse_loss': tf.constant(0.0, dtype=tf.float32), 'physical_loss': physical_loss}
        
    def train_step(self, data):
        def mse_branch():
            return self.mse_train_step(data)
    
        def physics_branch():
            return self.physics_train_step(data[:, 0])
    
        return tf.cond(tf.equal(tf.shape(data)[1], tf.constant(2)), mse_branch, physics_branch)

# 3. Generate training data
t_all = np.linspace(0, 30, 301).reshape(-1, 1)  # 3001 points from 0 to 30
# Exclude 0 and 30 seconds
mask = ~np.isin(t_all, [0, 30])
t_excluded = t_all[mask].reshape(-1, 1)
random_indices = np.random.choice(t_excluded.shape[0], 48, replace=False)  # Select 98 random indices
t_random = t_excluded[random_indices]
# Include 0 and 30 seconds
t_sim = np.vstack([[[0]], t_random, [[30]]])
y_sim = -4.9 * t_sim**2
t_col = np.linspace(0, 30, 301).reshape(-1, 1)

# 4. Train the model
model = FreeFallPINN()

# First, train with MSE
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
data = np.hstack((t_sim, y_sim))
mse_history = model.fit(data, epochs=20000, batch_size=32, verbose=0)

# Next, train with physical loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
physics_history = model.fit(t_col, epochs=10000, batch_size=32, verbose=0)

# 5. Plot loss functions over epochs
plt.figure(figsize=(10, 6))
plt.plot(mse_history.history['mse_loss'], label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE Loss over epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.squeeze(physics_history.history['physical_loss']), label='Physical Loss')  # Use np.squeeze here
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Physical Loss over epochs')
plt.legend()
plt.grid(True)
plt.show()

# 6. Generate a comparison plot of the true values and test data
t_test = np.linspace(0, 30, 300).reshape(-1, 1)
y_pred = model.predict(t_test)
y_true = -4.9 * t_test**2

plt.figure(figsize=(10, 6))
plt.plot(t_test, y_true, label="True")
plt.plot(t_sim,y_sim,'x',label="Observed")
plt.plot(t_test, y_pred, '--', label="Predicted")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
