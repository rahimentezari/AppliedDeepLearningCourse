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
    saver = tf.train.Saver()
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(args.path_model)
    print("args.path_model", args.path_model)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    print("model_files", model_files)
    # itrs = sorted(model_files.keys())
    # print("itrs", itrs)
    # itrs = [itrs[-1]]
    # print("itrs_last", itrs)

    # model_files = {1999: u'logs/model/itr-1999', 2999: u'logs/model/itr-2999', 3999: u'logs/model/itr-3999',
    #                4999: u'logs/model/itr-4999', 5999: u'logs/model/itr-5999', 6999: u'logs/model/itr-6999',
    #                7999: u'logs/model/itr-7999', 8999: u'logs/model/itr-8999', 9999: u'logs/model/itr-9999'
    #                }
    # model_files = {9999: u'logs/model/itr-9999'
    #                }
    # itrs = sorted(model_files.keys()) ## all checkpoint
    itrs = [sorted(model_files.keys())[10]]
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
        print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
        # print(result) this will print accuracy along with the tf.equal(label, output_class)
        print(_evaluate(
                model, saver, model_files[itr], sess,
                dataset, args.batch_size))
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc))*100))

    # # print test outputs
    # sample_input_generator = dataset.generate_sample(mode='test')
    # keys_sample = ['input', 'label']
    # batch_sample = {key: [] for key in keys_sample}
    # for i in range(100):
    #     try:
    #         example = next(sample_input_generator)
    #         for key in keys_sample:
    #             batch_sample[key].append(example[key])
    #     except StopIteration:
    #         empty = True
    # for key in keys_sample:
    #     batch_sample[key] = np.stack(batch_sample[key])
    # feed_dict = {}
    # feed_dict.update({model.inputs[key]: batch_sample[key] for key in keys_sample})
    # feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
    # print("feed_dict", feed_dict)
    # print("input_label", sess.run([model.inputs['label']], feed_dict))
    # print("prediction_all", sess.run([model.outputs['prediction']], feed_dict))
    # print("sparsity", sess.run([model.sparsity], feed_dict))
    #
    # test sample input
    print("|================== Samples From Test Data ==================|")
    sample_input_generator = dataset.generate_sample(mode='test')
    keys_sample = ['input', 'label']
    batch_sample = {key: [] for key in keys_sample}
    for i in range(100):
        try:
            example = next(sample_input_generator)
            # #print examples from test data
            # print("example", example)
            # print("example_shape", example['input'].shape)
            # print("example_sum", example['input'].sum())
            for key in keys_sample:
                batch_sample[key].append(example[key])
        except StopIteration:
            empty = True
    for key in keys_sample:
        batch_sample[key] = np.stack(batch_sample[key])
    # feed_dict = {}
    # feed_dict.update({model.inputs[key]: batch_sample[key] for key in keys_sample})
    # feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
    # result = sess.run([model.outputs], feed_dict)
    # print("input_label", sess.run([model.inputs['label']], feed_dict))
    # print("predicted", result[0]['prediction'])
    # print("sparsity", sess.run([model.sparsity], feed_dict))

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

    random_seed = 42
    # random_seed = np.random.seed(42)# if use this, error with Segmentation(line290): TypeError: an integer is required
    explainer = lime.lime_image.LimeImageExplainer(feature_selection='auto', random_state=random_seed)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=10, ratio=0.2, random_seed=random_seed) # https://kite.com/python/docs/skimage.segmentation.quickshift
    # # How segmentaion used in purturbation : we want 100 purturbted images. For each of these 100 images, we use a [0,1] mask to create such image. 100 * num_superpixels

    # print("batch_sample['label'][0]", batch_sample['label'][0])
    # print("batch_sample['input'][0]", batch_sample['input'][0].shape)

    def explain(instance, predict_fn, labels, **kwargs):
        return explainer.explain_instance(instance, predict_fn, labels, random_seed, **kwargs)

    mse_diff = []
    ssim = []
    for i in range(100):
        explanation = explain(batch_sample['input'][i], predict_proba, labels=(1,), top_labels=10,
                              num_features=10, num_samples=100, batch_size=100, distance_metric='cosine',
                              model_regressor=None, random_seed=42, segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(batch_sample['label'][i], num_features=10,
                                                    positive_only=True, hide_rest=True)

        mnist_mu = 0.13066062
        mnist_sigma = 0.30810776
        non_normallized_temp = (temp * mnist_sigma) + mnist_mu
        non_normallized_temp = np.absolute((temp * mnist_sigma) + mnist_mu)
        # print("non_normallized_temp", non_normallized_temp)
        # print("non_normallized_temp", non_normallized_temp.min(), non_normallized_temp.max())
        # print("non_normallized_temp", non_normallized_temp.min(), non_normallized_temp.max())
        # print("mask_min", mask.min())

        # plt.imsave('./Output_LIME/Ks_4_Md_10_ratio_0.2_Ns100_Tl10_Nf10_SNIP50_10KTrain.png', mark_boundaries(non_normallized_temp, mask))
        lime_output = mark_boundaries(non_normallized_temp, mask)
        mse_diff.append(mse(batch_sample['input'][i], lime_output))
        print("mse_diff", mse_diff)
        ssim.append(measure.compare_ssim(batch_sample['input'][i], lime_output, multichannel=True))
        print("ssim", ssim)
    mean_mse = statistics.mean(mse_diff)
    mean_ssim = statistics.mean(ssim)
    std_mse = statistics.stdev(mse_diff)
    std_ssim = statistics.stdev(ssim)

    print("mean_mse", mean_mse)
    print("mean_ssim", mean_ssim)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

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
