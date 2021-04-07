def visualize_voc_unet ():
    from data.voc2012_loader_segmentation import PascalVOCSegmentation
    from torch.utils.data.dataloader import DataLoader
    from visualize.visualize import visualize
    from models.unet import UNet

    dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=16, shuffle=False, num_workers=0)
    model = UNet(outputs=21, name='voc_unet')
    model.load()
    visualize(model, dataloader, model.name + '_visualization/')