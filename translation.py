from __future__ import absolute_import
import tensorflow as tf
import tensorlayer as tl
import numpy as np

import time
import os
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
from utils import *
from model import *


#====#====#====#====#====#====#====#====#====#
#====#====#====# File prepare #====#====#====#
#====#====#====#====#====#====#====#====#====#
timestr = time.strftime("%m%d_%H%M")
store_dir = "./Model" + timestr + '/'
sample_dir = store_dir + "samples"
save_dir = store_dir + "MNIST" + timestr
tl.files.exists_or_mkdir(sample_dir)
tl.files.exists_or_mkdir(store_dir)
# load_dir = "./Model0403_1138/"
load_dir = store_dir

#====#====#====#====#====#====#====#====#===#
#====#====#====# Hyper param #====#====#====#
#====#====#====#====#====#====#====#====#===#
is_train = False
is_joint = False
critic_iteration = 5
unsup_rate = 0
gpu_fraction = 0.7
rel_num = int(np.ceil(unsup_rate/(1-unsup_rate)))

lr = 2e-4
n_epoch = 50
batch_size = 16
image_size = 28
print_freq = 200

z_dim = 512
embedding_dim = 512
margin = 0.1

gan_type = 'wgan' #wgan-gp/gan/wgan
uni_type = ['loss_E','GRL']  #GRL/domainGAN/loss_E/loss_R
opt_type = 'adam' # adam/rmsprop

## log
file = open(os.path.join(store_dir, 'options.txt'), 'w')
file.write('')
file.write('Framework attributes: \n\
    Image translation tasks\n\
    correct domainGAN mistake \n\
    correct GRl mistake \n\
    add matching aware in unlb opt\n\
    change the matching aware\n\
    correct the tile problem\n\
    wgan-gp but no gploss test\n\
    correct the same training data in d_update iteration\n\
    g_loss optimize encoder?\n\
    a) GRL\n\
    wgan-gp apply\n\
    loss formu change (-)\
    Add domain classifer adversarial training \n\
    b) GAN\n\
    WGAN and clip \n')
file.write('=========== \n')
file.write('\n is_train '+str(is_train))
file.write('\n is_joint '+str(is_joint))
file.write('\n gan_type '+str(gan_type))
file.write('\n uni_type '+str(uni_type))
file.write('\n opt_type '+str(opt_type))
file.write('\n critic_iteration '+str(critic_iteration))
file.write('\n unsup_rate '+str(unsup_rate))
file.write('\n lr '+str(lr))
file.write('\n margin '+str(margin))
file.write('\n n_epoch '+str(n_epoch))
file.write('\n print_freq '+str(print_freq))
file.write('\n batch_size '+str(batch_size))
file.write('\n z_dim '+str(z_dim))
file.write('\n embedding_dim '+str(embedding_dim))
file.write('\n image_size '+str(image_size))
file.close

def main(unused_argv):

    #===#===#===#=== Loading Data #===#===#===#===
    # unlb_w_i,w_t, w_i, t_i, t_t, unlab_t_i, test_i, test_t, eval_i, eval_t = load_data(
    #     unsup_rate)
    # Load MNIST-M
    unlb_w_i,w_t, w_i, t_i, t_t, unlab_t_i, test_i, test_t, eval_i, eval_t = load_data2(
        unsup_rate)

    val_sample_t = t_t[50:66,:,:,:]
    save_images(val_sample_t, [int(np.sqrt(batch_size)), int(np.sqrt(batch_size))],
                        os.path.join(sample_dir, 'samplesss.png'))
    save_images(t_i[:16], [int(np.sqrt(batch_size)), int(np.sqrt(batch_size))],
                        os.path.join(sample_dir, 'target.png'))
    
    # #===#===#===#=== Placeholder #===#===#===#===
    l = tf.placeholder('float32', [])
    z = tf.placeholder('float32', shape=[batch_size, z_dim], name='noise')
    z2 = tf.placeholder('float32', shape=[batch_size, z_dim], name='noise2')
    # text = tf.placeholder('float32', shape=[batch_size, 1], name='text')
    # text_wrong = tf.placeholder('float32', shape=[batch_size, 1], name='text_wrong')
    # image_labeled = tf.placeholder('float32', [batch_size, image_size, image_size, 1], name='image_labeled')
    # image_wrong = tf.placeholder('float32', [batch_size, image_size, image_size, 1], name='image_wrong')

    text = tf.placeholder('float32', [batch_size, image_size, image_size, 1], name='text')
    text_wrong = tf.placeholder('float32', [batch_size, image_size, image_size, 1], name='text_wrong')
    image_labeled = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='image_labeled')
    image_wrong = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='image_wrong')
    image_unlabeled = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='image_unlabeled')
    image_unlabeled_wrong = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='image_unlabeled_wrong')
    
    _, ebd_text = image_encoder(text)
    _, ebd_t_wrong = image_encoder(text_wrong, reuse=True)
    # _, ebd_src = image_encoder(image_source)
    # _, ebd_lab = image_encoder2(image_labeled)
    # _, ebd_unlab = image_encoder2(image_unlabeled, reuse=True)

    #====#====#====#====#====#====#====#====#===#
    #===#===#===#=== Build Graph #===#===#===#===
    #====#====#====#====#====#====#====#====#===#
    ## sup part ##
    # _, ebd_text = text_encoder(text)
    # _, ebd_t_wrong = text_encoder(text_wrong, reuse=True)

    _, ebd_real = image_encoder2(image_labeled)
    _, ebd_i_wrong = image_encoder2(image_wrong, reuse=True)
    
    _, fake_img = image_decoder2(tf.concat([ebd_text, z], 1))

    _, d_real = discriminator2(image_labeled, ebd_text)
    _, d_fake = discriminator2(fake_img, ebd_text, reuse=True)
    # _, d_wrong = discriminator(image_wrong, ebd_text, reuse=True)
    _, d_wrong = discriminator2(image_labeled, ebd_t_wrong, reuse=True)

    ## unsup part ##
    _, ebd_unlb = image_encoder2(image_unlabeled, reuse=True)
    _, ebd_unlb_wrong = image_encoder2(image_unlabeled_wrong, reuse=True)

    _, fake_unlb_img = image_decoder2(tf.concat([ebd_unlb, z2], 1), reuse=True)

    _, d_unlb_real = discriminator2(image_unlabeled, ebd_unlb, reuse=True)
    _, d_unlb_fake = discriminator2(fake_unlb_img, ebd_unlb, reuse=True)
    _, d_unlb_wrong = discriminator2(image_unlabeled, ebd_unlb_wrong, reuse=True)    

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#=== Cost fomula #===#===#===#===
    #===#===#===#===#===#===#===#===#===#===#===#
    

    if gan_type == 'wgan-gp':
        ## Sup loss ##
        d_loss = tf.reduce_mean(d_fake+d_wrong)/2 - tf.reduce_mean(d_real)
        g_loss = -tf.reduce_mean(d_fake) 
        ## Unsup loss ##
        d_unlb_loss = -tf.reduce_mean(d_unlb_real - d_unlb_fake)
        g_unlb_loss = -tf.reduce_mean(d_unlb_fake)

        d_loss += wgangp_loss(image_labeled,fake_img,batch_size,ebd_text)
        d_unlb_loss += wgangp_loss(image_unlabeled,fake_unlb_img,batch_size,ebd_unlb)

    if gan_type == 'gan':
        rr = tl.cost.sigmoid_cross_entropy(d_real,
                                           tf.ones_like(d_real), name='drr')
        ff = tl.cost.sigmoid_cross_entropy(d_fake,
                                           tf.zeros_like(d_fake), name='dff')
        fr = tl.cost.sigmoid_cross_entropy(d_fake,
                                           tf.ones_like(d_fake), name='dfr')
        wf = tl.cost.sigmoid_cross_entropy(d_wrong,
                                           tf.ones_like(d_fake), name='dwf')

        un_rr = tl.cost.sigmoid_cross_entropy(d_unlb_real,
                                              tf.ones_like(d_unlb_real), name='drr')
        un_ff = tl.cost.sigmoid_cross_entropy(d_unlb_fake,
                                              tf.zeros_like(d_unlb_fake), name='dff')
        un_fr = tl.cost.sigmoid_cross_entropy(d_unlb_fake,
                                              tf.ones_like(d_unlb_fake), name='dfr')
        d_loss = rr + (ff + wf) / 2
        g_loss = fr

        d_unlb_loss = un_rr + un_ff
        g_unlb_loss = un_fr

    if gan_type == 'wgan':
        ## Sup loss ##
        d_loss = tf.reduce_mean(d_fake+d_wrong)/2 - tf.reduce_mean(d_real)
        g_loss = -tf.reduce_mean(d_fake) 
        ## Unsup loss ##
        d_unlb_loss = tf.reduce_mean(d_unlb_fake+ d_unlb_wrong)/2 - tf.reduce_mean(d_unlb_real)
        g_unlb_loss = -tf.reduce_mean(d_unlb_fake)

    if 'loss_E' in uni_type:
        norm_t = tf.nn.l2_normalize(ebd_text, 1)
        norm_i = tf.nn.l2_normalize(ebd_real, 1)
        pred_score = tf.matmul(norm_i, norm_t, transpose_b=True)
        diagonal = tf.diag_part(pred_score)
        cost_s = tf.maximum(0., margin - diagonal + pred_score)
        cost_im = tf.maximum(0., margin - tf.reshape(diagonal, [-1, 1]) + pred_score)
        cost_s = tf.multiply(cost_s, (tf.ones([tf.shape(norm_t)[0], tf.shape(norm_t)[0]]) - tf.eye(tf.shape(norm_t)[0])))
        cost_im = tf.multiply(cost_im, (tf.ones([tf.shape(norm_t)[0], tf.shape(norm_t)[0]]) - tf.eye(tf.shape(norm_t)[0])))
        loss_E = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)
        Unify_loss = loss_E

    if 'loss_R' in uni_type:
        x = tf.nn.l2_normalize(ebd_text, 1)
        v = tf.nn.l2_normalize(ebd_real, 1)
        x_w = tf.nn.l2_normalize(ebd_t_wrong, 1)
        v_w = tf.nn.l2_normalize(ebd_i_wrong, 1)
        loss_R = tf.reduce_mean(tf.maximum(0., margin - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
            tf.reduce_mean(tf.maximum(0., margin - cosine_similarity(x, v) + cosine_similarity(x_w, v)))
        Unify_loss = loss_R
    
    if 'loss_R_GRL' in uni_type:
        x = flip_gradient(ebd_text, l)
        v = flip_gradient(ebd_real, l)
        x_w = flip_gradient(ebd_t_wrong, l)
        v_w = flip_gradient(ebd_i_wrong, l)
        loss_R = tf.reduce_mean(tf.maximum(0., margin - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
            tf.reduce_mean(tf.maximum(0., margin - cosine_similarity(x, v) + cosine_similarity(x_w, v)))
        g_loss += loss_R


    if 'GRL' in uni_type:
        d_label_text = np.tile([1.,0.],[batch_size,1])
        d_label_imag = np.tile([0.,1.],[batch_size,1])
        domain_logits_text = domain_classifer(ebd_text,l,mode='GRL')
        domain_logits_imag = domain_classifer(ebd_real,l,mode='GRL',reuse=True)
        domain_loss_text = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logits_text, labels=d_label_text)
        domain_loss_imag = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logits_imag, labels=d_label_imag)
        g_loss += tf.reduce_mean(domain_loss_text + domain_loss_imag)

    if 'domainGAN' in uni_type:
        domain_logits_text = domain_classifer(ebd_text,mode='domainGAN')
        domain_logits_imag = domain_classifer(ebd_real,mode='domainGAN',reuse=True)
        dc_vars = tl.layers.get_variables_with_name('domain_classifer', True, True)

        domain_tt = tl.cost.sigmoid_cross_entropy(domain_logits_text,
                                            tf.ones_like(domain_logits_text), name='domain_tt')
        domain_ti = tl.cost.sigmoid_cross_entropy(domain_logits_text,
                                            tf.zeros_like(domain_logits_text), name='domain_ti')
        domain_it = tl.cost.sigmoid_cross_entropy(domain_logits_imag,
                                            tf.ones_like(domain_logits_text), name='domain_it')
        domain_ii = tl.cost.sigmoid_cross_entropy(domain_logits_imag,
                                            tf.zeros_like(domain_logits_text), name='domain_ii')

        domain_loss_d = domain_tt + domain_ii
        domain_loss_g_text = domain_ti
        domain_loss_g_imag = domain_it

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#=== Define train ops #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    ie_vars = tl.layers.get_variables_with_name('image_encoder', True, True)
    te_vars = tl.layers.get_variables_with_name('text_encoder', True, True)
    id_vars = tl.layers.get_variables_with_name('image_decoder', True, True)

    if opt_type == 'adam':
        g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5,beta2=0.9
                                    ).minimize(g_loss, var_list=id_vars+te_vars+ie_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9
                                    ).minimize(d_loss, var_list=d_vars)
        g_unlb_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5,beta2=0.9
                                    ).minimize(g_unlb_loss, var_list=id_vars)
        d_unlb_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9
                                    ).minimize(d_unlb_loss, var_list=d_vars)
        
        if ('loss_R' in uni_type) or ('loss_E' in uni_type):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!')
            u_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9
                                    ).minimize(Unify_loss, var_list=te_vars+ie_vars)
        else:
            u_optim = tf.constant(0,name='nothing')

    if opt_type == 'rmsprop':
        d_optim = tf.train.RMSPropOptimizer(lr).minimize(
            d_loss, var_list=d_vars)
        g_optim = tf.train.RMSPropOptimizer(lr).minimize(
            g_loss, var_list=id_vars+te_vars+ie_vars)
        d_unlb_optim = tf.train.RMSPropOptimizer(lr).minimize(
            d_unlb_loss, var_list=d_vars)
        g_unlb_optim = tf.train.RMSPropOptimizer(lr).minimize(
            g_unlb_loss, var_list=id_vars)
        if ('loss_R' in uni_type) or ('loss_E' in uni_type):
            
            u_optim = tf.train.RMSPropOptimizer(lr).minimize(
                Unify_loss, var_list=te_vars+ie_vars)
        else:
            u_optim = tf.constant(0,name='nothing')

    if 'domainGAN' in uni_type:
        domain_class_optim = tf.train.RMSPropOptimizer(lr).minimize(domain_loss_d, var_list=dc_vars)
        domain_text_optim = tf.train.RMSPropOptimizer(lr).minimize(domain_loss_g_text, var_list=te_vars)
        domain_imag_optim = tf.train.RMSPropOptimizer(lr).minimize(domain_loss_g_imag, var_list=ie_vars)

    if gan_type == 'wgan':
        d_clamp_op = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01))for var in d_vars]
    else:
        d_clamp_op = tf.constant(0,name='nothing')

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#===     Summarys     #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    true_sum  = tf.summary.histogram("d_real", d_real)
    fake_sum  = tf.summary.histogram("d_fake", d_fake)
    wrong_sum = tf.summary.histogram("d_wrong", d_wrong)

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    if 'loss_E' in uni_type:
        loss_e_sum = tf.summary.scalar("loss_E", loss_E)
    if 'loss_R' in uni_type:
        loss_r_sum = tf.summary.scalar("loss_R", loss_R)

    all_sum = tf.summary.merge_all()

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#===   Saving thing   #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    sess = tf.Session(config=config)
    tl.layers.initialize_global_variables(sess)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(store_dir+"logs", sess.graph)
    ### debug###
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    # sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#===  loading thing   #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    counter = 0
    if is_train == False :
        could_load, checkpoint_counter = load_ckpt(sess,saver,load_dir)
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS" + str(counter))
        else:
          print(" [!] Load failed...")

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===# loading validation network #===#==#
    #===#===#===#===#===#===#===#===#===#===#===#
    # x = tf.placeholder('float32',[batch_size, image_size, image_size, 1], name='x')
    # xx = tf.reshape(x, shape=[-1, 784])
    # y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # # define the network
    # network = tl.layers.InputLayer(xx, name='input')
    # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    # network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    # network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    # network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')

    # cost = tl.cost.cross_entropy(network.outputs, y_, name='xentropy')
    # correct_prediction = tf.equal(tf.argmax(network.outputs, 1), y_)
    # acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # d = np.load('mnist.npz', encoding = 'latin1')
    # lp = d['params']
    # tl.files.assign_params(sess, lp, network)

    val_sample_t = eval_t[50:66,:,:,:]
    # save_images(val_sample_t, [int(np.sqrt(batch_size)), int(np.sqrt(batch_size))],
    #                     os.path.join(sample_dir, 'samplesss.png'))
    # [[0], [1], [2], [3], [4], [5], [6],
    #                  [7], [8], [9], [0], [1], [2], [3], [4], [5]]
    val_sample_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#===  Training        #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    if is_train:
        tf.get_default_graph().finalize()
        begin_time = time.time()
        n_steps_per_epoch = int(t_i.shape[0] / batch_size)
        print(n_steps_per_epoch)

        for epoch in range(n_epoch):
            idx = 0  # data mark
            un_idx = 0  # unlabel data mark
            step = 0
            start_time = time.time()
            for i_train, t_train in tl.iterate.minibatches(t_i, t_t, batch_size, shuffle=False):

                # Sup training iteration
                p = float(step) / n_steps_per_epoch
                l_cal =  2. / (1. + np.exp(-10. * p)) - 1
                batch_z_1 = np.random.normal(loc=0.0, scale=1.0,size=(batch_size, z_dim)).astype(np.float32)
                feed_dict = {l:l_cal,
                             z: batch_z_1, 
                             text: t_train, 
                             image_labeled: i_train,
                             image_wrong: w_i[idx:idx+batch_size],
                             text_wrong: w_t[idx:idx+batch_size]}

                idx += batch_size
                counter += 1
                step += 1
                

                #### update Unify network ####
                if epoch < 20 :
                    for i in range(5):
                        sess.run([u_optim], feed_dict=feed_dict)
                else:
                    sess.run([u_optim], feed_dict=feed_dict)

                if 'domainGAN' in uni_type:
                    ##### update G ####
                    sess.run([g_optim,domain_text_optim,domain_imag_optim], feed_dict=feed_dict)
                    
                    ##### update D ####
                    if (counter+1) % print_freq == 0:
                        summary,_,_,_ = sess.run([all_sum, domain_class_optim, d_optim, d_clamp_op], feed_dict=feed_dict)
                        writer.add_summary(summary, counter)
                        print("Epoch %d / %d : Iteration %d took %.2fs . Total cost %.2fs" %
                              (epoch+1, n_epoch, counter, (time.time()-start_time), (time.time()-begin_time)))
                    else:
                        sess.run([d_optim,domain_class_optim,d_clamp_op], feed_dict=feed_dict)

                    # ##### Extra update D ####
                    # if counter <50 or counter%200==1 :
                    #     for i in range(critic_iteration):
                    #         sess.run([d_optim,d_clamp_op], feed_dict=feed_dict)
                else:
                    ##### update G ####
                    sess.run([g_optim], feed_dict=feed_dict)

                    ##### update D ####
                    if (counter+1) % print_freq == 0:
                        summary,_,_ = sess.run([all_sum, d_optim, d_clamp_op], feed_dict=feed_dict)
                        writer.add_summary(summary, counter)
                        print("Epoch %d / %d : Iteration %d took %.2fs . Total cost %.2fs" %
                              (epoch+1, n_epoch, counter, (time.time()-start_time), (time.time()-begin_time)))
                    else:
                        sess.run([d_optim, d_clamp_op], feed_dict=feed_dict)
                    
                    # ##### Extra update D ####
                    # if counter <50 or counter%200==1 :
                    #     for i in range(critic_iteration-1):
                    #         sess.run([d_optim,d_clamp_op], feed_dict=feed_dict)

                if is_joint:
                    for idx in range(rel_num):
                        batch_z_2 = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
                        feed_dict2 = {l:l_cal,
                                      z2: batch_z_2,
                                      image_unlabeled: unlab_t_i[un_idx:un_idx+batch_size],
                                      image_unlabeled_wrong: unlb_w_i[un_idx:un_idx+batch_size]}
                        un_idx = un_idx+batch_size
                        if un_idx < unlab_t_i.shape[0]:
                            ##### update G ####
                            sess.run([g_unlb_optim], feed_dict=feed_dict2)
                            ##### update D ####
                            sess.run([d_unlb_optim,d_clamp_op], feed_dict=feed_dict2)
                            # ##### Extra update D ####
                            # if counter <50 or counter%200==1 :
                            #     for i in range(critic_iteration):
                            #         sess.run([d_unlb_optim,d_clamp_op], feed_dict=feed_dict2)
                    


            #===#===#===#===#===#===#===#===#===#===#===#
            #===#===#===#===  Validation      #===#===#=#
            #===#===#===#===#===#===#===#===#===#===#===#  
            ## save samples ###
            fake = sess.run(fake_img, feed_dict={z: val_sample_z,text: val_sample_t})  
            save_images(fake, [int(np.sqrt(batch_size)), int(np.sqrt(batch_size))],
                        os.path.join(sample_dir, 'epoch_{:02d}.png'.format(epoch)))
            
            # ## valadation accuracy ###
            # eval_loss, eval_acc, n_batch = 0, 0, 0
            # for eval_image, eval_text in tl.iterate.minibatches(eval_i, eval_t, batch_size, shuffle=True):
            #     batch_z_1 = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
            #     feed_dict = {z: batch_z_1, text: eval_text}
            #     fake = sess.run(fake_img, feed_dict=feed_dict)

            #     dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
            #     feed_dict = {x: fake, y_: eval_text[:, 0]}
            #     feed_dict.update(dp_dict)
            #     err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            #     eval_loss += err
            #     eval_acc += ac
            #     n_batch += 1
            # print("   validation loss: %f" % (eval_loss / n_batch))
            # print("   validation acc: %f" % (eval_acc / n_batch))

            #===#===#===#===#===#===#===#===#===#===#===#
            #===#===#===#===  Checkpoint      #===#===#=#
            #===#===#===#===#===#===#===#===#===#===#===# 
            if (epoch+1) % 10 == 1:
                saver.save(sess, save_dir, global_step=counter)
                print("Model saved in file: %s" % save_dir)

        saver.save(sess, save_dir, global_step=counter)
        print("Model saved in file: %s" % save_dir)

    #===#===#===#===#===#===#===#===#===#===#===#
    #===#===#===#===  Testing         #===#===#=#
    #===#===#===#===#===#===#===#===#===#===#===#
    # else:
    #     test_loss, test_acc, n_batch = 0, 0, 0
    #     for test_image, test_text in tl.iterate.minibatches(test_i, test_t, batch_size, shuffle=True):
    #         batch_z_1 = np.random.normal(loc=0.0, scale=1.0,
    #                                      size=(batch_size, z_dim)).astype(np.float32)
    #         feed_dict = {z: batch_z_1, text: test_text}
    #         fake = sess.run(fake_img, feed_dict=feed_dict)

    #         dp_dict = tl.utils.dict_to_one(
    #             network.all_drop)  # disable noise layers
    #         feed_dict = {x: fake, y_: test_text[:, 0]}
    #         feed_dict.update(dp_dict)
    #         err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    #         test_loss += err
    #         test_acc += ac
    #         n_batch += 1
    #     print("   test loss: %f" % (test_loss / n_batch))
    #     print("   test acc: %f" % (test_acc / n_batch))

    sess.close()


if __name__ == "__main__":
    tf.app.run()










