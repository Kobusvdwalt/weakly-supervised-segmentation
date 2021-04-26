def visualize_voc_wass ():
    from data.voc2012_loader_segmentation import PascalVOCSegmentation
    from torch.utils.data.dataloader import DataLoader
    from visualize.visualize import visualize
    from models.wass import WASS

    dataloader = DataLoader(PascalVOCSegmentation('val'), batch_size=32, shuffle=False, num_workers=6)
    model = WASS(name='voc_wass')
    model.load()
    visualize(model, dataloader, model.name + '_visualization/')