from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        #import pdb;pdb.set_trace()
        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
        train_op = train(total_loss, global_step, total_num_examples)
    import pdb;pdb.set_trace()
    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(replica_grads, global_step, opt):
       
        average_grads = builder.average_gradients(replica_grads)
        apply_gradient_op = opt.apply_gradients(average_grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
             return tf.no_op(name='train')
 

    builder = ModelBuilder(devices[-1])
    global_step = builder.ensure_global_step()
    opt = configure_optimizer(global_step, total_num_examples)

    replica_images = tf.split(images, len(devices[:-1]))
    replica_labels = tf.split(labels, len(devices[:-1]))
    replica_grads = []
    replica_net = []
    replica_logits = []
    replica_total_loss = []

    for i in range(len(devices[:-1])):
        with tf.device(devices[i]):
              net, logits, total_loss = alexnet_inference(builder, replica_images[i], replica_labels[i], num_classes)
              replica_net.append(net)
              replica_logits.append(logits)
              replica_total_loss.append(total_loss)
              with tf.control_dependencies([total_loss]):
                       replica_grads.append( opt.compute_gradients(total_loss) )
    with tf.device(devices[-1]):       
         train_op = train(replica_grads, global_step, opt)
         net_stack = tf.stack(replica_net)
         net = tf.reduce_mean(net_stack, 0)
         logits_stack = tf.stack(replica_logits)
         logits = tf.reduce_mean(logits_stack, 0)
         total_loss_stack = tf.stack(replica_total_loss)
         total_loss = tf.reduce_mean(total_loss_stack, 0)

    return net, logits, total_loss, train_op, global_step
