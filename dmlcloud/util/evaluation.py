def accuracy(pred, label):
    if pred.shape[1] == 1:
        return ((pred > 0).float() == label).float().mean()
    else:
        return (pred.argmax(dim=1) == label).float().mean()


def top5_error(pred, label):
    top5_indices = pred.topk(5, dim=1)[1]
    top5_error = 1 - (top5_indices == label.unsqueeze(1)).float().max(dim=1)[0].mean()
    return top5_error
