# -*- coding:utf-8 -*-
# !/usr/bin/env python

'''
############################################################
rename tensorflow variable.
############################################################
'''

import tensorflow as tf
import os
import re


def load_model(model_path, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model_path)
    if (os.path.isfile(model_exp)):
        print('not support: %s' % model_exp)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    return saver


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def rename(dict, oldckpt_dir, newckpt_path, add_prefix=''):
    '''rename tensorflow variable, just for checkpoint file format.'''

    replace_from = [key for key in dict]
    replace_to = [dict[key] for key in replace_from]

    assert len(replace_from) == len(replace_to)

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(oldckpt_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(oldckpt_dir, var_name)

            # Set the new name
            new_name = var_name

            for index in range(len(replace_from)):
                new_name = new_name.replace(replace_from[index], replace_to[index])

            # if add_prefix:
            #     new_name = add_prefix + new_name

            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)

        # Save the variables
        saver = load_model(oldckpt_dir)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, newckpt_path)


if __name__ == '__main__':
    # the code comes from https://blog.csdn.net/qq_33666011/article/details/80522564

    # the content of the oldckpt_dir:
    #   checkpoint
    #   old.data-00000-of-00001
    #   old.index
    #   old.meta

    # the including of the checkpoint file:
    # model_checkpoint_path: "ep164-step45000-loss0.001"
    # all_model_checkpoint_paths: "ep164-step45000-loss0.001"


    # old_name:new_name
    dict = {'conv1_1/kernel': 'conv_11/kernel',
            'conv1_1/bias': 'conv_11/bias'}
    oldckpt_dir = './ckpt/'
    newckpt_path = './new'
    rename(dict, oldckpt_dir, newckpt_path)
