from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .vggcommon import vgg_part_conv, vgg_inference, vgg_loss, vgg_eval


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = vgg_inference(builder, images, labels, num_classes)

        if not is_train:
            return vgg_eval(net, labels)

        global_step = builder.ensure_global_step()
        # Compute gradients
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)

    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    def train(replica_grads, global_step, opt, builder):

        average_grads = builder.average_gradients(replica_grads)
        apply_gradient_op = opt.apply_gradients(average_grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
             return tf.no_op(name='train')


    builder = ModelBuilder(devices[-1])
    global_step = builder.ensure_global_step()
    #opt = configure_optimizer(global_step, total_num_examples)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    replica_images = tf.split(images, len(devices[:-1]))
    replica_labels = tf.split(labels, len(devices[:-1]))
    replica_grads = []
    replica_net = []
    replica_logits = []
    replica_total_loss = []

    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
      for i in range(len(devices[:-1])):
        with tf.device(devices[i]):
              net, logits, total_loss = vgg_inference(builder, replica_images[i], replica_labels[i], num_classes)
              replica_net.append(net)
              replica_logits.append(logits)
              replica_total_loss.append(total_loss)
              with tf.control_dependencies([total_loss]):
                       replica_grads.append( opt.compute_gradients(total_loss) )
    with tf.device(devices[-1]):
         train_op = train(replica_grads, global_step, opt, builder)
         net_stack = tf.stack(replica_net)
         net = tf.reduce_mean(net_stack, 0)
         logits_stack = tf.stack(replica_logits)
         logits = tf.reduce_mean(logits_stack, 0)
         total_loss_stack = tf.stack(replica_total_loss)
         total_loss = tf.reduce_mean(total_loss_stack, 0)

    return net, logits, total_loss, train_op, global_step
