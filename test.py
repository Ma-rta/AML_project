import os
import torch
import torch.utils.data
from model.triplet_match.model import TripletMatch

from DomainToText.dataset import EvalDataset, split
from DomainToText.evaluation import do_eval

def test():
    _, _, test_list = split('/content/AML_project/datalabels/')
    testset = EvalDataset(test_list)

    model = TripletMatch()
    model.cuda()
    model.eval()

    if os.path.exists('/content/AML_project/finetuned/BEST_checkpoint1.pth'):
        model.load_state_dict(torch.load('/content/AML_project/finetuned/BEST_checkpoint1.pth'), strict=False)

    with torch.no_grad():
        i2p_result, p2i_result = do_eval(model, testset)
        eval_metric = i2p_result + p2i_result
        print(f'mAP on test set: {eval_metric:.3f}')

if __name__ == '__main__':
    test()