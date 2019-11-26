import os
import tensorflow as tf
import numpy as np

from helpers import cache_json
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import lime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(args, model, sess, dataset):
    print('|========= START TEST =========|')
    saver = tf.train.Saver(max_to_keep=10)
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(args.path_model)
    print("args.path_model", args.path_model)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    print("model_files", model_files)
    itrs = sorted(model_files.keys())
    print("itrs", itrs)
    itrs = [itrs[-1]]
    print("itrs_last", itrs)
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
        print(result)
        print(_evaluate(
                model, saver, model_files[itr], sess,
                dataset, args.batch_size))
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc))*100))

    # print test outputs
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
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch_sample[key] for key in keys_sample})
    feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
    print("feed_dict", feed_dict)
    print("input_label", sess.run([model.inputs['label']], feed_dict))
    print("prediction_all", sess.run([model.outputs['prediction']], feed_dict))
    print("sparsity", sess.run([model.sparsity], feed_dict))

    # test sample input
    sample_input_generator = dataset.generate_sample(mode='test')
    # sample_input = next(sample_input)
    # print("sample_input", sample_input)

    keys_sample = ['input', 'label']
    batch_sample = {key: [] for key in keys_sample}
    for i in range(10):
        try:
            example = next(sample_input_generator)
            for key in keys_sample:
                batch_sample[key].append(example[key])
        except StopIteration:
            empty = True
    for key in keys_sample:
        batch_sample[key] = np.stack(batch_sample[key])
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch_sample[key] for key in keys_sample})
    feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
    result = sess.run([model.outputs], feed_dict)
    print("logits", result[0]['logits'])
    print("predicted", result[0]['prediction'])
    # for x in batch_sample['input']:
    #     print("sum", x.sum(), x.shape)
    # for x in batch_sample['label']:
    #     print("label", x)
    print("sparsity", sess.run([model.sparsity], feed_dict))

    def predict_proba(X):
        sample_input_generator = dataset.generate_sample(mode='test')
        # sample_input = next(sample_input)
        # print("sample_input", sample_input)

        keys_sample = ['input', 'label']
        batch_sample = {key: [] for key in keys_sample}
        # for i in range(100):
        #     try:
        #         example = next(sample_input_generator)
        #         for key in keys_sample:
        #             batch_sample[key].append(example[key])
        #     except StopIteration:
        #         empty = True
        # for key in keys_sample:
        #     batch_sample[key] = np.stack(batch_sample[key])
        feed_dict = {}
        # feed_dict.update({model.inputs[key]: batch_sample[key] for key in keys_sample})
        feed_dict.update({model.inputs['input']: X})
        feed_dict.update({model.compress: False, model.is_train: False, model.pruned: True})
        print("feed_dict", feed_dict)
        result = sess.run([model.outputs], feed_dict)
        return result[0]['logits']

    random_seed = 42
    # random_seed = np.random.seed(42)# if use this, error with Segmentation(line290): TypeError: an integer is required
    random_state = 42
    explainer = lime.lime_image.LimeImageExplainer(feature_selection='auto', random_state=random_seed)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2, random_seed=random_seed)

    print("batch_sample['label'][0]", batch_sample['label'][0])
    print("batch_sample['input'][0]", batch_sample['input'][0].shape)

    def explain(instance, predict_fn, labels, **kwargs):
        return explainer.explain_instance(instance, predict_fn, labels, random_seed, **kwargs)

    explanation = explain(batch_sample['input'][0], predict_proba, labels=(1,), top_labels=10,
                          num_features=10, num_samples=1, batch_size=100, distance_metric='cosine',
                          model_regressor=None, random_seed=42)
    # temp, mask = explanation.get_image_and_mask(batch_sample['label'][0], num_features=10,
    #                                             positive_only=True, hide_rest=True)
    # plt.imsave('./Output_LIME/MNIST_org_LIME_Ns1K_Tl1_Nf5_SNIP99_1KTrain.png', mark_boundaries(temp / 255.0, mask))



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
