# Age estimation New method (Still working)

## Idea

* Deep metric learning
* L1 distance between features
* Image distribution from training dataset

## Method (Still need to work)

 1. Calculate the standardization per image after convert to gray scale

    | step 1                                                       |
    | ------------------------------------------------------------ |
    | img = tf.io.read_file(im)<br/>    img = tf.image.decode_jpeg(img)<br/>    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])<br/>    img = tf.image.rgb_to_grayscale(img)<br/>    img = tf.image.per_image_standardization(img) |

    

 2. Gather the images per classes after 1 step

    | step 2                                                       |
    | ------------------------------------------------------------ |
    | for i in range(len(tr_img)):<br/>        imgs, labs = next(it)<br/>        age_buf[labs.numpy()] += imgs[0, ..., 0]<br/>        age_count[labs.numpy()] += 1<br/>        if i % 100 == 0:<br/>            print(i) |

    

 3. After 2 step, The age_buf shape are [classes, FLAGS.img_size, FLAGS.img_size]. And then calculate the PCA with age_buf (output of the PCA saved in final_buf)

    | step 3                                                       |
    | ------------------------------------------------------------ |
    | final_pca = np.zeros(shape=[FLAGS.classes, 100], dtype=np.float32)<br/>    for j in range(FLAGS.classes):<br/><br/>        age_buf[j] = age_buf[j] / age_count[j]<br/>        pca = PCA(n_components=100)<br/>        img_pca = pca.fit(age_buf[j])<br/>        eigen_value_img_pca = img_pca.explained_variance_<br/>        std = tf.math.reduce_std(eigen_value_img_pca)<br/>        mean = tf.reduce_mean(eigen_value_img_pca)<br/><br/>        for i in range(100):<br/>            final_pca[j][i] = tf.exp( -(eigen_value_img_pca[i] - mean)**2 / (2.*std*std) ) |

    

 4. Set model output same as final_pca's shape ( [batch, classes, 100] )

 5.  Calculate the distances between model output and final_pca. And apply to increase and decrease loss

    | step 5                                                       |
    | ------------------------------------------------------------ |
    | with tf.GradientTape() as tape:<br/>        logits = run_model(model, images, True) # batch x 100<br/>        in_loss = 0.<br/>        de_loss = 0.<br/>        loss = 0.<br/>        for i in range(FLAGS.batch_size):<br/>            logits_ = logits[i]<br/>            labels_ = labels[i]<br/><br/>            distance = tf.reduce_mean(tf.abs(logits_ - input_pca), 1)<br/>            #less_distance = tf.reduce_min(distance) # 최소 거리값은 구했다!<br/>            distance_arg = tf.argmin(distance)<br/>            label_arg = (tf.argmin(labels_)).numpy()<br/><br/>            decrease_loss = 0.<br/>            for j in range(FLAGS.classes):<br/>                if distance[label_arg] != distance[j]:<br/>                    decrease_loss += (-distance[j]-1)/(1 - tf.math.exp(0.2*distance[j]) + 0.000001)  # 유사도가 낮은것 !!<br/>                else:<br/>                    increas_loss = tf.math.exp(distance[label_arg] - 1.5)       # 유사도가 높은것 !!<br/><br/>            loss += (decrease_loss + increas_loss) / FLAGS.classes  # 이게 지금 loss감소가 아이에 이뤄지지 않는다<br/><br/>        loss /= FLAGS.batch_size<br/>    grads = tape.gradient(loss, model.trainable_variables)<br/>    optim.apply_gradients(zip(grads, model.trainable_variables)) |

## Problem

* Loss function is not stable (or unchangeable)
* **Need to find out the new loss**

