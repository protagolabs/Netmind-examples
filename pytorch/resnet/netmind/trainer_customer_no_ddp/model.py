import torchvision.models as models

def get_model(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model_name_or_path))
        model = models.__dict__[args.model_name_or_path](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model_name_or_path))
        model = models.__dict__[args.model_name_or_path]()

    return model
