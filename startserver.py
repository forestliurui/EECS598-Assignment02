"""
A simple script to start tensorflow servers with different roles.
"""
import os

import tensorflow as tf

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

#
tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_2proc = tf.train.ClusterSpec({
    "ps": [
        "localhost:2222"
    ],
    "worker": [
        "localhost:2223",
        "localhost:2224"
    ]
})

clusterSpec_2dev_2proc = tf.train.ClusterSpec({
    "ps": [
        "salat0:2222"
    ],
    "worker": [
        "salat1:2223",
        "salat1:2224"
    ]
})
clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node-0:2222"
    ],
    "worker" : [
        "node-1:2222",
        "node-2:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "node-0:2222"
    ],
    "worker" : [
        "node-1:2222",
        "node-2:2222",
        "node-3:2222",
        "node-4:2222"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    '1dev-2proc': clusterSpec_2proc,
    '2dev-2proc': clusterSpec_2dev_2proc,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
server.join()
