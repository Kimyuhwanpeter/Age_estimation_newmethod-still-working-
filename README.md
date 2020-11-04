# Age estimation New method (Still working)

## Idea

* Deep metric learning
* L1 distance between features
* Image distribution from training dataset

## Method (Still need to work)

 1. Calculate the standardization per image after convert to gray scale

        img = tf.io.read_file(im)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.per_image_standardization(img)

 2. Gather the images per classes after 1 step

        age_buf = np.zeros(shape=[FLAGS.classes, FLAGS.img_size, FLAGS.img_size], dtype=np.float32)
        age_count = np.zeros(shape=[FLAGS.classes], dtype=np.int32)
        for i in range(len(tr_img)):
            imgs, labs = next(it)
            age_buf[labs.numpy()] += imgs[0, ..., 0]
            age_count[labs.numpy()] += 1
            if i % 100 == 0:
                print(i)

 3. After 2 step, The age_buf shape are [classes, FLAGS.img_size, FLAGS.img_size]. And then calculate the PCA with age_buf (output of the PCA saved in final_buf)

        final_pca = np.zeros(shape=[FLAGS.classes, 100], dtype=np.float32)
        for j in range(FLAGS.classes):
        
            age_buf[j] = age_buf[j] / age_count[j]
            pca = PCA(n_components=100)
            img_pca = pca.fit(age_buf[j])
            eigen_value_img_pca = img_pca.explained_variance_
            std = tf.math.reduce_std(eigen_value_img_pca)
            mean = tf.reduce_mean(eigen_value_img_pca)
        
            for i in range(100):
                final_pca[j][i] = tf.exp( -(eigen_value_img_pca[i] - mean)**2 / (2.*std*std) )

 4. Set model output same as final_pca's shape ( [batch, classes, 100] )

 5.  Calculate the distances between model output and final_pca. And apply to increase and decrease loss

         with tf.GradientTape() as tape:
             logits = run_model(model, images, True)
             in_loss = 0.
             de_loss = 0.
             loss = 0.
             for i in range(FLAGS.batch_size):
                 logits_ = logits[i]
                 labels_ = labels[i]

                 distance = tf.reduce_mean(tf.abs(logits_ - input_pca), 1)
                 #less_distance = tf.reduce_min(distance)
                 distance_arg = tf.argmin(distance)
                 label_arg = (tf.argmin(labels_)).numpy()

                 decrease_loss = 0.
                 for j in range(FLAGS.classes):
                     if distance[label_arg] != distance[j]:
                         decrease_loss += (-distance[j]-1)/(1 - tf.math.exp(0.2*distance[j]) + 0.000001)
                     else:
                         increas_loss = tf.math.exp(distance[label_arg] - 1.5)

                 loss += (decrease_loss + increas_loss) / FLAGS.classes  

             loss /= FLAGS.batch_size
         grads = tape.gradient(loss, model.trainable_variables)
         optim.apply_gradients(zip(grads, model.trainable_variables))

## Problem

* Loss function is not stable (or unchangeable)
* **Need to find out the new loss**

