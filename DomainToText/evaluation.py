import numpy as np

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def compute_mAP(match_scores, gt_matrix, mode='i2p'):
    """
    INPUT:
    - match_scores: [img_num x phrase_num], match_scores[i,j] = cosine_sim(emb(img_i), emb(phrase_j)) (shows how well img_i and phrase_j matches)
    - gt_matrix: [img_num x phrase_num], gt_matrix[i,j] shows which img_i corresp. to which phrase_j (1 if they corresp., 0 otherwise)
    - mode: 'i2p' = retrieve images given phrases, 'p2i' = retrieve phrases given images
    """
    img_num = gt_matrix.shape[0]
    phrase_num = gt_matrix.shape[1]

    if mode == 'i2p':
        # each row is prediction for one image. phrase sorted by pred scores. values are whether the phrase is correct
        i2p_correct = np.zeros_like(gt_matrix, dtype=bool)  # img_num x phrase_num
        i2p_phrase_idxs = np.zeros_like(i2p_correct, dtype=int)
        for img_i in range(img_num):
            phrase_idx_sorted = np.argsort(-match_scores[img_i, :])
            i2p_phrase_idxs[img_i] = phrase_idx_sorted
            i2p_correct[img_i] = gt_matrix[img_i, phrase_idx_sorted]
        retrieve_binary_lists = i2p_correct
    elif mode == 'p2i':
        # each row is prediction for one prhase. images sorted by pred scores. values are whether the image is correct
        p2i_correct = np.zeros_like(gt_matrix, dtype=bool).transpose()  # class_num x img_num
        p2i_img_idxs = np.zeros_like(p2i_correct, dtype=int)
        for pi in range(phrase_num):
            img_idx_sorted = np.argsort(-match_scores[:, pi])
            p2i_img_idxs[pi] = img_idx_sorted
            p2i_correct[pi] = gt_matrix[img_idx_sorted, pi]
        retrieve_binary_lists = p2i_correct
    else:
        raise NotImplementedError
    
    # calculate mAP
    return mean_average_precision(retrieve_binary_lists)

def predict(out_img, out_txt, valset):
    match_scores = np.zeros((len(valset), valset))
    for i, img in enumerate(out_img):
        for j, phr in enumerate(out_txt):
            match_scores[i,j] = - np.sum(np.power(img - phr, 2)) # l2_s
    return match_scores
                

def do_eval (model, valset):
    out_img = model.img_encoder(valset.t_images.cuda()).cpu().numpy()
    out_txt = model.lang_encoder(valset.descr).cpu().numpy()
    gt_matrix = np.eye(len(valset))
    match_scores = predict(out_img, out_txt, len(valset))

    i2p_result = compute_mAP(match_scores, gt_matrix, mode='i2p')
    p2i_result = compute_mAP(match_scores, gt_matrix, mode='p2i') 
    return i2p_result, p2i_result

