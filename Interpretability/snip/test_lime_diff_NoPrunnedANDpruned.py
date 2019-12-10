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



def test(args, model, sess, dataset):
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


    def predict_proba(X):
        sample_input_generator = dataset.generate_sample(mode='test')
        # sample_input = next(sample_input)
        # print("sample_input", sample_input)

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
        # print("feed_dict", feed_dict)
        result = sess.run([model.outputs], feed_dict)
        return result[0]['logits']

    def explain(instance, predict_fn, labels, **kwargs):
        return explainer.explain_instance(instance, predict_fn, labels, random_seed, **kwargs)
    
    saver = tf.train.Saver()
    # ########################################################################################### ORG model (no pruning)
    args.path_summary = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
                                     'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/logs_001',
                                     'summary')
    args.path_model = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
                                   'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/logs_001',
                                   'model')
    args.path_assess = os.path.join('/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/'
                                    'Interpretability/LIME/AppliedDeepLearningCourse/Interpretability/snip/logs_001',
                                    'assess')

    state = tf.train.get_checkpoint_state(args.path_model)
    print("args.path_model", args.path_model)
    model_files_org = {int(s[s.index('itr') + 4:]): s for s in state.all_model_checkpoint_paths}
    print("model_files_org", model_files_org)

    itrs = [sorted(model_files_org.keys())[9]]  # ### acc = 99.12%
    print("itrs", itrs)

    # Subset of iterations.
    itr_subset = itrs
    assert itr_subset
    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        # run evaluate and/or cache
        result = cache_json(
            os.path.join(args.path_assess, dataset.datasource, 'itr-{}.json'.format(itr)),
            lambda: _evaluate(
                model, saver, model_files[itr], sess,
                dataset, args.batch_size),
            makedir=True)
        # print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
        # print(result) this will print accuracy along with the tf.equal(label, output_class)
        # print(_evaluate(
        #     model, saver, model_files_org[itr], sess,
        #     dataset, args.batch_size))
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc)) * 100))

    # ##################################################################################################### Pruned model
    state = tf.train.get_checkpoint_state(args.path_model)
    print("args.path_model", args.path_model)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    print("model_files", model_files)

    itrs = [sorted(model_files.keys())[0]]
    print("itrs", itrs)

    # Subset of iterations.
    itr_subset = itrs
    assert itr_subset
    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        # run evaluate and/or cache
        result = cache_json(
            os.path.join(args.path_assess, dataset.datasource, 'itr-{}.json'.format(itr)),
            lambda: _evaluate(
                model, saver, model_files[itr], sess,
                dataset, args.batch_size),
            makedir=True)
        # print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
        # print(result) this will print accuracy along with the tf.equal(label, output_class)
        # print(_evaluate(
        #         model, saver, model_files[itr], sess,
        #         dataset, args.batch_size))
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc))*100))


















    # random_seed = 42
    # # random_seed = np.random.seed(42)# if use this, error with Segmentation(line290): TypeError: an integer is required
    # explainer = lime.lime_image.LimeImageExplainer(feature_selection='auto', random_state=random_seed)
    # segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=10, ratio=0.2, random_seed=random_seed) # https://kite.com/python/docs/skimage.segmentation.quickshift
    # # # How segmentaion used in purturbation : we want 100 purturbted images. For each of these 100 images, we use a [0,1] mask to create such image. 100 * num_superpixels
    #

    #
    # for i in range(100):
    #     print("== == == == == purturbation started!== == == == ==")
    #     explanation = explain(batch_sample['input'][i], predict_proba, labels=(1,), top_labels=10,
    #                           num_features=10, num_samples=100, batch_size=100, distance_metric='cosine',
    #                           model_regressor=None, random_seed=42, segmentation_fn=segmenter)
    #     temp, mask = explanation.get_image_and_mask(batch_sample['label'][i], num_features=10, negative_only=True,
    #                                                 positive_only=False, hide_rest=True)
    #
    #
    #
    # random_seed = 42
    # # random_seed = np.random.seed(42)# if use this, error with Segmentation(line290): TypeError: an integer is required
    # explainer = lime.lime_image.LimeImageExplainer(feature_selection='auto', random_state=random_seed)
    # segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=10, ratio=0.2,
    #                                   random_seed=random_seed)  # https://kite.com/python/docs/skimage.segmentation.quickshift
    #
    # # # How segmentaion used in purturbation : we want 100 purturbted images. For each of these 100 images, we use a [0,1] mask to create such image. 100 * num_superpixels
    #
    # def explain(instance, predict_fn, labels, **kwargs):
    #     return explainer.explain_instance(instance, predict_fn, labels, random_seed, **kwargs)
    #
    # mse_diff = []
    # ssim = []
    # for i in range(100):
    #     explanation = explain(batch_sample['input'][i], predict_proba, labels=(1,), top_labels=10,
    #                           num_features=10, num_samples=100, batch_size=100, distance_metric='cosine',
    #                           model_regressor=None, random_seed=42, segmentation_fn=segmenter)
    #     temp_org, mask_org = explanation.get_image_and_mask(batch_sample['label'][i], num_features=10, negative_only=True,
    #                                                 positive_only=False, hide_rest=True)
    #
    # np.savetxt('./mask_lime_9/Ks_4_Md_10_ratio_0.2_Ns100_Tl10_Nf10_SNIP001_9KTrain_mask_HideRestF_negative_only_org.'
    #            'txt', mask_org)
    # np.savetxt('./mask_lime_9/Ks_4_Md_10_ratio_0.2_Ns100_Tl10_Nf10_SNIP992_10KTrain_mask_HideRestF_negative_only.txt',
    #            mask)
    #
    # mask_org_bool = mask_org.astype(bool)
    # mask_bool = mask.astype(bool)
    #
    # sum_diff = np.logical_xor(mask_org_bool, mask_bool).astype(int).sum()
    # print(sum_diff)


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

    results = { # has to be JSON serialiazable
        'accuracy': np.mean(accuracy).astype(float),
        'accuracy_all': accuracy,
        'num_example': len(accuracy),
    }
    assert results['num_example'] == dataset.dataset['test']['input'].shape[0]
    return results
