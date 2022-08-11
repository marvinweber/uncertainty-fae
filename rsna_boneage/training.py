from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from torchsummary import summary
from torchvision import transforms

from torchvision.models import resnet18, resnet34, inception_v3

from boneage.rsna_bone_net import resnet18 as resnet18_boneage, resnet34 as resnet34_boneage
from boneage.rsna_bone_litmodel import RSNABoneageLitModel
from boneage.rsna_bone_dataloading import RSNABoneageDataModule


def main():
    rsna_net = get_inception()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5)
    ])
    rsna_boneage_datamodule = RSNABoneageDataModule(
        annotation_file_train='./images/with_preprocessed_train_annotations.csv',
        annotation_file_val='./images/with_preprocessed_val_annotations.csv',
        annotation_file_test='./images/with_preprocessed_test_annotations.csv',
        batch_size=8,
        transform=transform,
        target_dimensions=(299, 299),
        rescale_boneage=True,
        rebalance_classes=True)
    rsna_boneage_datamodule.setup('fit')
    train_dataset = rsna_boneage_datamodule.dataset_train
    boneage_distribution = train_dataset.get_bonage_distribution()
    loss_weights = {iv: 1 + (1 / count) for iv, count in boneage_distribution.items()}

    model = RSNABoneageLitModel(net=rsna_net, loss_weights=None)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".")
    csv_logger = pl_loggers.CSVLogger(save_dir=".")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', save_top_k=-1, filename='{epoch}-{val_loss:2f}')
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', mode='min', patience=10)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [checkpoint_callback, early_stopping_callback]

    trainer = Trainer(accelerator='gpu', max_epochs=100, log_every_n_steps=50,
                      logger=[csv_logger, tb_logger], callbacks=callbacks)
    trainer.fit(model, datamodule=rsna_boneage_datamodule)


def get_resnet_18():
    resnet_pretrained = resnet18(pretrained=True, progress=True)
    resnet_pretrained_state_dict = resnet_pretrained.state_dict()

    resnet = resnet18_boneage(pretrained=False, progress=True, num_classes=1)
    resnet_state_dict = resnet.state_dict()

    for key in resnet_state_dict.keys():
        if key in resnet_pretrained_state_dict and 'fc' not in key:
            resnet_state_dict[key] = resnet_pretrained_state_dict[key]
    resnet.load_state_dict(resnet_state_dict)

    return resnet


def get_resnet_34():
    resnet_pretrained = resnet34(pretrained=True, progress=True)
    resnet_pretrained_state_dict = resnet_pretrained.state_dict()

    resnet = resnet34_boneage(pretrained=False, progress=True, num_classes=1)
    resnet_state_dict = resnet.state_dict()

    for key in resnet_state_dict.keys():
        if key in resnet_pretrained_state_dict and 'fc' not in key:
            resnet_state_dict[key] = resnet_pretrained_state_dict[key]
    resnet.load_state_dict(resnet_state_dict)

    return resnet


def get_inception():
    inception_pretrained = inception_v3(pretrained=True, progress=True)
    inception_pretrained_state_dict = inception_pretrained.state_dict()

    inception = inception_v3(pretrained=False, num_classes=1)
    inception_state_dict = inception.state_dict()

    for key in inception_state_dict.keys():
        if key in inception_pretrained_state_dict and 'fc' not in key:
            inception_state_dict[key] = inception_pretrained_state_dict[key]
    inception.load_state_dict(inception_state_dict)

    return inception


if __name__ == '__main__':
    main()
