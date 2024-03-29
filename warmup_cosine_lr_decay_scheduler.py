"""
@author sourabhxiii
"""

import numpy as np
import keras
from keras import backend as K

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update at the beginning of each stage 2: update messages at each step. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose == 1:
            if self.global_step == (self.warmup_steps + 1): print("\n Warm state ends.")
            if self.global_step == (self.hold_base_rate_steps + 1): print("\n Hold base LR state ends.")
        elif self.verbose == 2:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class TransferLearningLRScheduler(keras.callbacks.Callback):
    """ In case of Transfer Learning, we start by training only the added layers with a higher learning rate.
        This ensures that the newly added random weights are adjusted to the pre-trained convolutional weights.
        This burn-in period mitigates the risk of juggling around the convolutional weights at training start 
        and consequently slowing down the training process.
        After the burn-in period we should train all weights in the CNN with a low learning rate..
    """
    def __init__(self
                , learning_rate_initial=0.01
                , learning_rate_base=0.0001
                , burn_in_steps=5
                , layers = 0
                , global_step_init=0
                , verbose=1):
        super(TransferLearningLRScheduler, self).__init__()
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate_base = learning_rate_base
        self.burn_in_steps = burn_in_steps
        self.global_step = global_step_init
        self.layers = layers
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = 0.
        if self.global_step >= self.burn_in_steps:
            lr = self.learning_rate_base
            # make the entire network trainable
            for layer in self.model.layers[self.layers:]:
                layer.trainable = True
            self.model.compile(self.model.optimizer, self.model.loss, self.model.metrics)
        else:
            lr = self.learning_rate_initial
        assert lr != 0., 'LR can not be zero.'
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose == 1:
            if self.global_step == (self.burn_in_steps + 1):
                print("\n Burn-in state ends."
                      "Using LR = %s" %(lr))
        elif self.verbose == 2:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))



'''

https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/

# Create a model.
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Number of training samples.
sample_count = 12

# Total epochs to train.
epochs = 100

# Number of warmup epochs.
warmup_epoch = 10

# Training batch size, set small value here for demonstration purpose.
batch_size = 4

# Base learning rate after warmup.
learning_rate_base = 0.001

total_steps = int(epochs * sample_count / batch_size)

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size)

# Generate dummy data.
data = np.random.random((sample_count, 100))
labels = np.random.randint(10, size=(sample_count, 1))

# Convert labels to categorical one-hot encoding.
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=epochs, batch_size=batch_size,
          verbose=0, callbacks=[warm_up_lr])

import matplotlib.pyplot as plt
plt.plot(warm_up_lr.learning_rates)
plt.xlabel('Step', fontsize=20)
plt.ylabel('lr', fontsize=20)
plt.axis([0, total_steps, 0, learning_rate_base*1.1])
plt.xticks(np.arange(0, total_steps, 50))
plt.grid()
plt.title('Cosine decay with warmup', fontsize=20)
plt.show()

'''