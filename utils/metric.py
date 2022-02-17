#  Copyright (c) 2021. NavInfo EU
#  All rights reserved.
#
#  You may only use this code under the terms of the NavInfo license.
#

from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import torch


def rcnn_2_boxes(boxes):
    '''
    Converts RCNN labels to the common format that metrics (e.g. mAP) can understand
    '''
    new_boxes = []

    for i, pred in enumerate(boxes): # note: iterates over batches and composes all boxes from all images from batch into one list

        if 'scores' not in pred:
            pred['scores'] = torch.ones(pred['labels'].shape)
        boxes_per_batch = []
        for b, s, l in zip(pred['boxes'], pred['scores'], pred['labels']):
            sb = [i, int(l.cpu().numpy() if torch.is_tensor(l) else l),
                  float(s.cpu().numpy() if torch.is_tensor(s) else s)]
            sb.extend(list(b.cpu().numpy() if torch.is_tensor(b) else b))
            boxes_per_batch.append(sb)
        new_boxes.append(boxes_per_batch)
    return new_boxes

# rcnn format
# boxes tensor obj x 4, in pixels
# labels tensor obj, int
# scores tensor obj, int

# boxes format
# list of [img_idx_in_batch, label, score, coords (4)] all floats/ints

adapters = {
    ('RCNN', 'boxes'): rcnn_2_boxes
}


class Metric:
    __metaclass__ = ABCMeta

    """ A metric base class
    Attributes:
        target_format_metric  - can be used to set certain target format
        (for target adaptation by target adapters)
    """
    target_format_metric = None

    state = None

    @abstractmethod
    def _set_state(self, *args, **kwargs):
        return

    @abstractmethod
    def _reset_state(self, *args, **kwargs):
        return

    @abstractmethod
    def _get_state(self, *args, **kwargs):
        return

    def __init__(self, **params):
        """ Method that initializes the metric (per test). Incorporates .set()
        :param params: parameters and their values that come from GUI (or defaults from config json)
        """
        self.params = params
        self.target_format_model = params.get('target_format', None)
        self.init()

    def init(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute_batch(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        # todo extend to dict output (multiple values)
        """ Method that computes the metric value per batch. Equivalent to .update()
        :param args: two positional arguments are provided: targets, predictions
        :param kwargs: additional parameters such as: handler to dataset, model, and num_classes are provided
        :return:
            scalar, the metric value per batch (if meaningful), the value:
                can be ued in feedback loop for test parameter optimization,
                will be stored along with the intermediate data;
            None, if the metric value per batch is meaningless
        """
        raise NotImplementedError

    def verify(self, kwargs):
        """ Method for verifying parameters needed for the Metric class """
        pass

    def compute_final(self):
        """ Method that finalizes metric computation per test/dataset. Can incorporate .reset()
        :return:
            scalar, the metric value per per test/dataset
        """
        return None

    # decorators
    def predict(call):
        def wrapper(self, *args, **kwargs):
            def _predict(labels, expand=False):
                if labels.ndim > 1:
                    num_classes = kwargs.get('num_classes', None)
                    if labels.shape[1] == num_classes or num_classes is None:

                        labels = np.argmax(labels, axis=1)
                        if expand:
                            labels = np.expand_dims(labels, axis=1)
                return labels

            expand = np.array([a.ndim for a in args])
            expand = np.all(expand == expand[0])

            args = (_predict(a, expand=expand) for a in args)
            return call(self, *args, **kwargs)

        return wrapper

    # todo make it more generic with options to ret args or None
    def try_me(call):
        def wrapper(self, *args, **kwargs):
            try:
                return call(self, *args, **kwargs)
            except (RuntimeError, ValueError, TypeError, KeyError) as ex:
                _len = max([len(a) if hasattr(a, '__len__') else 1 for a in args])
                return [np.nan for _ in range(_len)]

        return wrapper

    # todo add selection convert to numpy or torch.tensor
    def format_labels(accept_arrays_only=True):
        def decorator(call):
            def wrapper(self, *args, **kwargs):

                def _is_array(labels):
                    for l in labels:
                        if not isinstance(l, (np.ndarray,
                                              torch.Tensor)):
                            return False
                    return True

                def _format_label(labels):
                    if torch.is_tensor(labels):
                        labels = labels.cpu().numpy()
                    return labels.copy()

                if accept_arrays_only and not _is_array(args):
                    return np.array([np.nan])
                args = (_format_label(a) for a in args)
                return call(self, *args, **kwargs)

            return wrapper

        return decorator

    def adapt_labels(call):
        def wrapper(self, *args, **kwargs):

            target_format_model = kwargs.pop('target_format',
                                             getattr(kwargs.get('model', None), 'target_format',
                                                     self.target_format_model))

            if self.target_format_metric is not None and target_format_model != self.target_format_metric:
                assert target_format_model is not None
                _adapter = adapters[target_format_model, self.target_format_metric]
                args = (_adapter(a) for a in args)
            return call(self, *args, **kwargs)

        return wrapper


class MetricWithSimpleState(Metric):

    def init(self):
        self.state = {}
        self._reset_state()

    def _set_state(self, value, key='state'):
        self.state[key].extend(value)

    def _reset_state(self, key='state'):
        self.state[key] = []

    def _get_state(self, key='state'):
        return self.state[key]

    def compute_final(self):
        state = self._get_state()
        return np.array(state).mean()