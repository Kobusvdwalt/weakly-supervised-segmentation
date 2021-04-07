def visualize_voc_vgg16 ():
    from data.voc2012_loader_segmentation import PascalVOCSegmentation
    from torch.utils.data.dataloader import DataLoader
    from visualize.visualize import visualize
    from models.vgg16 import Vgg16GAP

    dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=16, shuffle=False, num_workers=0)
    model = Vgg16GAP(outputs=21, name='voc_vgg16')
    model.load()
    visualize(model, dataloader, model.name + '_visualization/')