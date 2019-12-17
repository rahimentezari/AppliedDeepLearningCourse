import os
import tensorflow as tf
import numpy as np
import statistics
from helpers import cache_json
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage import measure
import lime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import cv2


def test(args, model, sess, dataset):
    sess.run(tf.global_variables_initializer())
    print('|========= START TEST =========|')
    # test sample input
    print("|================== Samples From Test Data ==================|")
    sample_input_generator = dataset.generate_sample(mode='test')
    keys_sample = ['input', 'label']
    batch_sample = {key: [] for key in keys_sample}
    for i in range(100):
        try:
            example = next(sample_input_generator)
            for key in keys_sample:
                batch_sample[key].append(example[key])
        except StopIteration:
            empty = True
    for key in keys_sample:
        batch_sample[key] = np.stack(batch_sample[key])

    def _evaluate(model, saver, model_file, sess, dataset, batch_size):

        # load model
        if saver is not None and model_file is not None:
            saver.restore(sess, model_file)
        else:
            raise FileNotFoundError
        # load test set; epoch generator
        generator = dataset.generate_example_epoch(mode='test')

        accuracy = []
        empty = False
        while not empty:
            # construct a batch of test examples
            keys = ['input', 'label']
            batch = {key: [] for key in keys}
            for i in range(batch_size):
                try:
                    example = next(generator)
                    for key in keys:
                        batch[key].append(example[key])
                except StopIteration:
                    empty = True
            # run the batch
            if batch['input'] and batch['label']:
                # stack and padding (if necessary)
                for key in keys:
                    batch[key] = np.stack(batch[key])
                feed_dict = {}
                feed_dict.update({model.inputs[key]: batch[key] for key in keys})
                feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
                result = sess.run([model.outputs], feed_dict)
                accuracy.extend(result[0]['acc_individual'])

        results = {  # has to be JSON serialiazable
            'accuracy': np.mean(accuracy).astype(float),
            'accuracy_all': accuracy,
            'num_example': len(accuracy),
        }
        assert results['num_example'] == dataset.dataset['test']['input'].shape[0]
        return results

    def predict_proba(X):
        sample_input_generator = dataset.generate_sample(mode='test')


        keys_sample = ['input', 'label']
        batch_sample_predict_proba = {key: [] for key in keys_sample}
        for i in range(100):
            try:
                example = next(sample_input_generator)
                for key in keys_sample:
                    batch_sample_predict_proba[key].append(example[key])
            except StopIteration:
                empty = True
        for key in keys_sample:
            batch_sample_predict_proba[key] = np.stack(batch_sample_predict_proba[key])
        feed_dict = {}
        # feed_dict.update({model.inputs[key]: batch_sample_predict_proba[key] for key in keys_sample})
        feed_dict.update({model.inputs['input']: X, model.inputs['label']:batch_sample_predict_proba['label']})
        feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})

        result = sess.run([model.outputs], feed_dict)
        return result[0]['logits']


    def predict_probability_softmax(X):
        sample_input_generator = dataset.generate_sample(mode='test')


        keys_sample = ['input', 'label']
        batch_sample_predict_proba = {key: [] for key in keys_sample}
        for i in range(100):
            try:
                example = next(sample_input_generator)
                for key in keys_sample:
                    batch_sample_predict_proba[key].append(example[key])
            except StopIteration:
                empty = True
        for key in keys_sample:
            batch_sample_predict_proba[key] = np.stack(batch_sample_predict_proba[key])
        feed_dict = {}
        # feed_dict.update({model.inputs[key]: batch_sample_predict_proba[key] for key in keys_sample})
        feed_dict.update({model.inputs['input']: X, model.inputs['label']:batch_sample_predict_proba['label']})
        feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})

        result = sess.run([model.outputs], feed_dict)
        return result[0]['logits_softmax']

    def explain(instance, predict_fn, labels, **kwargs):
        return explainer.explain_instance(instance, predict_fn, labels, random_seed, **kwargs)

    random_seed = 42
    explainer = lime.lime_image.LimeImageExplainer(feature_selection='auto', random_state=random_seed)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=10, ratio=0.2, random_seed=random_seed)
    # # https://kite.com/python/docs/skimage.segmentation.quickshift
    # # How segmentaion used in purturbation : we want 100 purturbted images.
    # # For each of these 100 images, we use a [0,1] mask to create such image. 100 * num_superpixels

    saver = tf.train.Saver()
    sum_mask_diff = []
    masks_org = []
    masks_pruned = []
    # for i in range(100):
    #     # ####################################################################################### ORG model (no pruning)
    #     args.path_summary = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
    #                                      'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/'
    #                                      'logs_0_org/summary')
    #     args.path_model = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
    #                                    'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/'
    #                                    'logs_0_org/model')
    #
    #     state = tf.train.get_checkpoint_state(args.path_model)
    #     model_files_org = {int(s[s.index('itr') + 4:]): s for s in state.all_model_checkpoint_paths}
    #
    #     itr = sorted(model_files_org.keys())[10] # ### acc = 99.12%
    #     saver.restore(sess, model_files_org[itr])
    #
    #     # ### Gets the explanation
    #     explanation = explain(batch_sample['input'][i], predict_proba, labels=(1,), top_labels=10,
    #                           num_features=10, num_samples=100, batch_size=100, distance_metric='cosine',
    #                           model_regressor=None, random_seed=42, segmentation_fn=segmenter)
    #     temp_org, mask_org = explanation.get_image_and_mask(batch_sample['label'][i], num_features=10, negative_only=True,
    #                                                 positive_only=False, hide_rest=True)
    #     masks_org.append(mask_org)
    #     mask_path = './logs_0_org'
    #     if not os.path.exists(mask_path):
    #         os.mkdir(mask_path)
    #     np.savetxt(
    #         mask_path + '/ORG_Ks_4_Md_10_ratio_0.2_Ns100_Tl10_Nf10_SNIP_Org_10KTrain_mask_HideRestF_negative_only' +
    #         str(batch_sample['label'][i]) + "_" + str(i) + '.txt',
    #         mask_org, fmt='%i')
    # with open(mask_path + '/masks_org.pk', 'wb') as fp:
    #     pickle.dump(masks_org, fp)
    # ##################################################################################################### Pruned model
    # for i in range(100):
    #     state = tf.train.get_checkpoint_state(args.path_model)
    #     model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    #     itr = sorted(model_files.keys())[10]
    #     saver.restore(sess, model_files[itr])
    #
    #
    #     # ### Gets the explanation
    #     explanation = explain(batch_sample['input'][i], predict_proba, labels=(1,), top_labels=10,
    #                           num_features=10, num_samples=100, batch_size=100, distance_metric='cosine',
    #                           model_regressor=None, random_seed=42, segmentation_fn=segmenter)
    #     temp, mask = explanation.get_image_and_mask(batch_sample['label'][i], num_features=10, negative_only=True,
    #                                                positive_only=False, hide_rest=True)
    #     masks_pruned.append(mask)
    # mask_path = './logs/mask_lime_10K'
    # if not os.path.exists(mask_path):
    #     os.mkdir(mask_path)
    # with open(mask_path + '/masks_pruned.pk', 'wb') as fp:
    #     pickle.dump(masks_pruned, fp)

    # ########################################################################################orginal trained model (2D)
    args.path_summary = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
                                     'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/'
                                     'logs_0_org_2d/summary')
    args.path_model = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
                                   'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/'
                                   'logs_0_org_2d/model')

    state = tf.train.get_checkpoint_state(args.path_model)
    model_files_org = {int(s[s.index('itr') + 4:]): s for s in state.all_model_checkpoint_paths}

    itr = sorted(model_files_org.keys())[10]
    saver.restore(sess, model_files_org[itr])

    with open(mask_path + '/masks_org.pk', 'rb') as fp:
        masks_org = pickle.load(fp)

    a = predict_probability_softmax(mask)
    print(a)
    #     masks_pruned.append(mask)
    #
    #     mask_path = './logs/mask_lime_10K'
    #     if not os.path.exists(mask_path):
    #         os.mkdir(mask_path)
    #     # np.savetxt(
    #     #     mask_path + '/Pruned_Ks_4_Md_10_ratio_0.2_Ns100_Tl10_Nf10_SNIP50_10KTrain_mask_HideRestF_negative_only' +
    #     #     str(batch_sample['label'][i]) + "_" + str(i) + '.txt',
    #     #     mask, fmt='%i')
    #
    # # ### compare the explanations for only mask
    # with open(mask_path + '/masks_org.pk', 'rb') as fp:
    #     masks_org = pickle.load(fp)
    # for x, y in zip(masks_org, masks_pruned):
    #     mask_org_bool = x.astype(bool)
    #     mask_bool = y.astype(bool)
    #     one_sample_diff = np.logical_xor(mask_org_bool, mask_bool).astype(int).sum()
    #     sum_mask_diff.append(one_sample_diff)
    # print("LIST_sum_mask_diff", sum_mask_diff)
    # mean = statistics.mean(sum_mask_diff)
    # std = statistics.stdev(sum_mask_diff)
    # print("mean, std", mean, std)
    # # remove bad LIME Org
    # bad_org_list = [3, 13, 55, 43, 77, 18, 51, 90, 4, 6, 42, 48, 49, 8, 15, 23, 45, 53, 59, 17, 36, 70, 79, 86, 97, 12,
    #                 16, 62]
    # sum_mask_diff_removedBad = []
    # for i in range(100):
    #     if i not in bad_org_list:
    #         sum_mask_diff_removedBad.append(sum_mask_diff[i])
    #
    # #
    # # for x in sum_mask_diff:
    # #     if x.index() not in bad_org_list:
    # #         sum_mask_diff_removedBad.append(x)
    # print(len(sum_mask_diff_removedBad))
    # mean = statistics.mean(sum_mask_diff_removedBad)
    # std = statistics.stdev(sum_mask_diff_removedBad)
    # print("mean, std", mean, std)







