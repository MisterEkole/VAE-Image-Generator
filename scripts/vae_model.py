
from tkinter import _Padding
import tensorflow as tf
import keras
from keras import layers

import numpy as np

'''
Building the VAE(Variational Autoencoder), this network consist of 3 parts:
An encoder layer that turns a real image into a mean and a variance in a latent space
A sampling layer that takes  such mean and variance  and use them to sample random point from latent space
A decoder layer that turns points from the latent space back to into images

Model subclassing implementation
'''

#Latent space sampling layer

class Sampler(layers.Layer):
    def call(self,inputs):
        z_mean,z_log_var=inputs
        batch_size=tf.shape(z_mean)[0]
        z_size=tf.shape(z_mean)[1]
        epsilon=tf.random.normal(shape=(batch_size, z_size)) #generating a batch of normal random vectors
        return z_mean +tf.exp(0.5*z_log_var)*epsilon #applying the VAE Sampling formula


#encoder layer
class encoder(layers.Layer):
    '''
    Maps image inputs to (z_mean,z_log_var,z)
    '''
    def __init__(self, latent_dim=32,inter_dim=64,name='encoder',**kwargs):
        super(encoder,self).__init__(name=name, *kwargs)
        self.dense1=layers.Conv2D(inter_dim,3,activation='relu', strides=2, padding='same')
        self.dense2=layers.Conv2D(latent_dim,3,activation='relu',strides=2, padding='same')
        self.dense3=layers.Flatten()
        self.dense4=layers.Dense(16, activation='relu')
        self.z_mean= layers.Dense(latent_dim)
        self.z_log_var=layers.Dense(latent_dim)
        self.sampling=Sampler()
    def call(self, inputs):
        x=self.dense1(inputs)
        y=self.dense2(x)
        m=self.dense3(y)
        n=self.dense4(m)
        z_mean=self.z_mean(n)
        z_log_var=self.z_log_var(n)
        z=self.sampling((z_mean,z_log_var))
        
        return z_mean, z_log_var, z
    
#decoder layer
class decoder(layers.Layer):
    def __init__(self, original_dim, inter_dim=64, name='decoder', **kwargs):
        super(decoder, self).__init__(name=name, **kwargs)
        self.dense1=layers.Conv2DTranspose(inter_dim,3,activation='relu', strides=2, padding='same')
        self.dense2=layers.Conv2DTranspose(32,3, activation='relu', strides=2, padding='same')
        self.d_outputs=layers.Conv2D(original_dim,3, activation='sigmoid', padding='same')
        
    def call(self, inputs):
        x=self.dense1(inputs)
        m=self.dense2(x)
        n_output=self.d_outputs(m)
        return n_output
    
    
    
'''Stacking up the whole VAE Model with sublayers'''

class VAE(keras.Model):
    '''Combination of encoder and decoder into an end to end model for training'''
    def __init__(
        self,
        original_dim,
        inter_dim=64,
        latent_dim=32,
        name='autoencoder',
        **kwargs
    ):
        super(VAE,self).__init__(name=name, **kwargs)
        self.original_dim=original_dim
        self.encoder=encoder(latent_dim=latent_dim,inter_dim=inter_dim)
        self.decoder=decoder(original_dim, inter_dim=inter_dim)
        
    def call(self, inputs):
        z_mean,z_log_var,z=self.encoder(inputs)
        reconstructed=self.decoder(z)
      
         # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
    
    
    
    
    
    
    
    
    
    
        
        
        