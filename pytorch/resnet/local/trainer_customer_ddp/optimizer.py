from torch.optim import SGD

def get_optimizer(model, args):
    # Prepare optimizer
    print('setup optimizer...')
    optimizer = SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    return optimizer