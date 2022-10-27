import torchvision.models as models

def get_model(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model_name_or_path))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model_name_or_path))
        model = models.__dict__[args.arch]()

    return model
