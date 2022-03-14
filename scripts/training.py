import vae_model
import tensorflow as tf
import numpy as np

'''
Training Loop
'''

original_dim=784
vae=vae_model.VAE(original_dim,64,32)

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn=tf.keras.losses.MeanSquaredError()
loss_metric=tf.keras.metrics.Mean()

'''
Loading the MNIST Dataset
'''
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)


epochs=20

#iterate over epochs

for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
            
            
'''
Generating sample images
'''
import matplotlib.pyplot as plt

n=30
img_size=28
figure=np.zeros((img_size*n,img_size*n))

x_grid=np.linspace(-1,1,n)
y_grid=np.linspace(-1,1,n)[::-1]

for i, yi in enumerate(y_grid):
  for j,xi in enumerate(x_grid):
    z_sample=np.array([[xi,yi]])
    x_decoded=vae.decoder.predict(z_sample)
    img=x_decoded[0].reshape(img_size, img_size)

    figure[
           i * img_size: (i+1)*img_size,
           j* img_size: (j+1)*img_size,
    ]= img


plt.figure(figsize=(10,10))
start_range= img_size//2
end_range= n* img_size+ start_range

pix_range=np.arange(start_range, end_range, img_size)
x_sample_range=np.round(x_grid,1)
y_sample_range=np.round(y_grid,1)

plt.xticks(pix_range,x_sample_range)
plt.yticks(pix_range,y_sample_range)

plt.xlabel("Z[0]")
plt.ylabel("z[1]")

plt.imshow(figure, cmap='Greys_r')
