class Config:
    def __init__(self,
        # Classifier
        classifier_dataset_root: str = '/',
        classifier_name: str = 'name',
        classifier_epochs: int = 1,
        classifier_batch_size_train: int = 8,
        classifier_batch_size_cams: int = 8,
        classifier_image_size: int = 448,
        classifier_pretrained: bool = False,
        classifier_pretrained_unfreeze: int = 0,

        # Cams
        cams_save_gt_labels: bool = False,
        
        # AffinityNet
        affinity_net_batch_size: int = 8,

        # Segmentation
        semseg_name: str = 'name',
        semseg_batch_size = 4,
        semseg_dataset_root = '/',
        semseg_image_size = 513,
        semseg_epochs = 5,
        semseg_pretrained = False,
    ):
        self.classifier_dataset_root = classifier_dataset_root
        self.classifier_name = classifier_name
        self.classifier_epochs = classifier_epochs
        self.classifier_batch_size_train = classifier_batch_size_train
        self.classifier_batch_size_cams = classifier_batch_size_cams
        self.classifier_image_size = classifier_image_size
        self.classifier_pretrained = classifier_pretrained
        self.classifier_pretrained_unfreeze = classifier_pretrained_unfreeze

        self.cams_save_gt_labels = cams_save_gt_labels

        self.affinity_net_batch_size = affinity_net_batch_size

        self.semseg_name = semseg_name
        self.semseg_batch_size = semseg_batch_size
        self.semseg_dataset_root = semseg_dataset_root
        self.semseg_image_size = semseg_image_size
        self.semseg_epochs = semseg_epochs
        self.semseg_pretrained = semseg_pretrained
        

    def toDictionary(self):
        return {
            'classifier_dataset_root': self.classifier_dataset_root,
            'classifier_name': self.classifier_name,
            'classifier_epochs': self.classifier_epochs,
            'classifier_batch_size_train': self.classifier_batch_size_train,
            'classifier_batch_size_cams': self.classifier_batch_size_cams,
            'classifier_image_size': self.classifier_image_size,
            'classifier_pretrained': self.classifier_pretrained,
            'classifier_pretrained_unfreeze': self.classifier_pretrained_unfreeze,

            'cams_save_gt_labels': self.cams_save_gt_labels,

            'affinity_net_batch_size': self.affinity_net_batch_size,

            'semseg_name': self.semseg_name,
            'semseg_batch_size': self.semseg_batch_size,
            'semseg_dataset_root': self.semseg_dataset_root,
            'semseg_image_size': self.semseg_image_size,
            'semseg_epochs': self.semseg_epochs,
            'semseg_pretrained': self.semseg_pretrained
        }


class ConfigManager():
    def __init__(self):
        self.setConfig(Config())
    
    def getConfig(self) -> Config:
        return self.config

    def setConfig(self, config):
        self.config = config

config_manager = ConfigManager()