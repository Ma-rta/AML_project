def test():
    _, _, test_list = split('/content/AML_project/datalabels/')
    testset = EvalDataset(test_list)

    model = TripletMatch()
    model.cuda()
    model.eval()

    if os.path.exists('/content/finetuned/BEST_checkpoint.pth'):
        model.load_state_dict(torch.load('/content/finetuned/BEST_checkpoint.pth'), strict=False)

    with torch.no_grad():
        i2p_result, p2i_result = do_eval(model, testset)
        eval_metric = i2p_result + p2i_result
        print(f'mAP on test set: {eval_metric:.3f}')

if __name__ == '__main__':
    test()