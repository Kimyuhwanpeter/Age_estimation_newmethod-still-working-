# Age estimation New method (Still working)

## Idea

* Deep metric learning
* L1 distance between features
* Image distribution from training dataset

## Method (Still need to work)

 1. Calculate the standardization per image after convert to gray scale

    | step 1                                                       |
    | ------------------------------------------------------------ |
    | img = tf.io.read_file(im)<br/>    img = tf.image.decode_jpeg(img)<br/>    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])<br/>    img = tf.image.rgb_to_grayscale(img)    img = tf.image.per_image_standardization(img) |


## Problem

* Loss function is not stable (or unchangeable)
* **Need to find out the new loss**

