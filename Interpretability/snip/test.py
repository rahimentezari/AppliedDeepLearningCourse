import os
import tensorflow as tf
import numpy as np

from helpers import cache_json


def test(args, model, sess, dataset):
    print('|========= START TEST =========|')
    saver = tf.train.Saver(max_to_keep=20)
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(args.path_model)
    print("args.path_model", args.path_model)
    for s in state.all_model_checkpoint_paths:
        print(s)
    model_files = {int(s[s.index('itr')+4:]): s for s in state.all_model_checkpoint_paths}
    print("model_files", model_files)
    itrs = sorted(model_files.keys())
    print("itrs", itrs)

    # # if max_to_keep=None only uses the last checkpoint, we set max_to_keep=20
    # model_files = {1999: u'logs/model/itr-1999', 2999: u'logs/model/itr-2999', 3999: u'logs/model/itr-3999',
    #                4999: u'logs/model/itr-4999', 5999: u'logs/model/itr-5999', 6999: u'logs/model/itr-6999',
    #                7999: u'logs/model/itr-7999', 8999: u'logs/model/itr-8999', 9999: u'logs/model/itr-9999',
    #                }
    # itrs = sorted(model_files.keys())
    # print("itrs", itrs)

    # Subset of iterations.
    itr_subset = itrs
    assert itr_subset
    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        # run evaluate and/or cache
        print(os.path.join(args.path_assess, dataset.datasource, 'itr-{}.json'.format(itr)))
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
        'tf.equal(label, output_class)': accuracy,
        'num_example': len(accuracy),
    }
    assert results['num_example'] == dataset.dataset['test']['input'].shape[0]
    return results
