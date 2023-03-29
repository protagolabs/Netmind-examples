# Train Resnet with Trainer

## local training
* model training
```bash
python train_netmind.py --data="../../datasets/tiny-imagenet-200/train/" --val_data="../../datasets/tiny-imagenet-200/val/"
```

* model testing

```bash
python test_netmind.py --val_data="../../datasets/tiny-imagenet-200/val/"
```