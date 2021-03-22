import numpy as np
# import torch
import pandas as pd


def jaccard_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)

    localizations_a: list of localizations
    localizations_a: list of localizations

    return:
        (ndarray)
    """
    size_set_A = len(localizations_a)
    size_set_B = len(localizations_b)

    localizations_a_start = [localization[0] for localization in localizations_a]
    localizations_a_end = [localization[1] for localization in localizations_a]

    localizations_b_start = [localization[0] for localization in localizations_b]
    localizations_b_end = [localization[1] for localization in localizations_b]

    localizations_a_start = np.array([localizations_a_start for _ in range(size_set_B)])
    localizations_a_end = np.array([localizations_a_end for _ in range(size_set_B)])

    localizations_b_start = np.transpose(np.array([localizations_b_start for _ in range(size_set_A)]))
    localizations_b_end = np.transpose(np.array([localizations_b_end for _ in range(size_set_A)]))

    length_a = localizations_a_end - localizations_a_start
    length_b = localizations_b_end - localizations_b_start


    # intersection
    max_min = np.maximum(localizations_a_start, localizations_b_start)
    min_max = np.minimum(localizations_a_end, localizations_b_end)
    intersection = np.maximum((min_max - max_min), 0)

    try:
        overlaps = intersection / (length_a + length_b - intersection)
    except:
        print('here')
    return overlaps


def extract_events_from_binary_mask(binary_mask, fs=1):
    binary_mask = np.array([0] + binary_mask.tolist() + [0])
    diff_data = np.diff(binary_mask)
    starts = np.where(diff_data == 1)[0] / fs
    ends = np.where(diff_data == -1)[0] / fs

    assert len(starts) == len(ends)
    events = []
    for i, _ in enumerate(starts):
        events += [(starts[i], ends[i])]

    return events


def compute_f1_score(event_pred, event_true):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for idx in range(event_pred.shape[0]):
        tp, fp, fn = compute_tp_fp_fn_for_each_entry(event_pred[idx], event_true[idx])
        total_tp += tp
        total_fp += fp
        total_fn +=fn

    if (total_fp + total_fn)==0:
        precision = 0
    else:
        precision = total_tp / (total_tp + total_fp)

    if (total_tp + total_fn) == 0:
        recall = 0
    else:
        recall = total_tp / (total_tp + total_fn)

    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return f1_score



def compute_tp_fp_fn_for_each_entry(prediction, reference, min_iou=0.3):
    """takes 2 event scorings
        (in array format [[start1, end1], [start2, end2], ...])
        and outputs the f1 score.

        Parameters
        ----------
        min_iou : float
            minimum intersection-over-union with a true event to be considered
            a true positive.
        """
    if len(prediction) == 0:
        return 0, 0, len(reference)
    if len(reference) == 0:
       return 0, len(prediction), 0

    iou = jaccard_overlap(prediction, reference)

    # Number of true events which are matched by a predicted events
    TP1 = np.sum(np.amax((iou >= min_iou), axis=0))
    # Number of predicted events which match a true event
    TP2 = np.sum(np.amax((iou >= min_iou), axis=1))

    # In order to avoid duplicate, a true event can only match a true events and a predicted event
    true_positive = min(TP1, TP2)
    false_positive = len(prediction) - true_positive
    false_negative = len(reference) - true_positive

    return true_positive, false_positive, false_negative



def format_predictions_for_scoring(mask, window_length=100):
    """
    Stack events in one single list for F1 computation
    :param mask:
    :param window_length:
    :return:
    """
    result = []
    for i, elt in enumerate(mask):
        events = extract_events_from_binary_mask(elt)
        if len(events) > 0:
            events = [[i * window_length + start, i * window_length + end] for start, end in events]
        else:
            events = []
        result.append(events)
    return np.array(result)


def dreem_sleep_apnea_custom_metric(y_pred, y_true):
    event_true = format_predictions_for_scoring(np.array(y_pred))
    event_pred = format_predictions_for_scoring(np.array(y_true))
    return compute_f1_score(event_pred, event_true)


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_Y_TRUE = '--------.csv'
    CSV_FILE_Y_PRED = '--------.csv'
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(dreem_sleep_apnea_custom_metric(df_y_true, df_y_pred))
