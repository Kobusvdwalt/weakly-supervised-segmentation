class Config:
    def __init__(self,
        sweep_id: str = '0',
        eval_dataset_root: str = '/',

        # Classifier
        classifier_dataset_root: str = '/',
        classifier_name: str = 'name',
        classifier_epochs: int = 1,
        classifier_batch_size_train: int = 8,
        classifier_image_size: int = 448,
        classifier_pretrained: bool = False,
        classifier_pretrained_unfreeze: int = 0,

        # Cams
        cams_produce_batch_size: int = 32,
        cams_measure_batch_size: int = 64,
        cams_save_gt_labels: bool = False,
        cams_bg_alpha: int = 16,
        cams_bg_alpha_low: int = 4,
        cams_bg_alpha_high: int = 32,
        
        # AffinityNet
        affinity_net_name: str = 'affinitynet',
        affinity_net_image_size: int = 448,
        affinity_net_batch_size: int = 8,
        affinity_net_epochs: int = 6,
        affinity_net_gamma: int = 5,                    # Pixel range for affnet 
        affinity_net_log_t: int = 8,                    # Matmul count
        affinity_net_bg_alpha: int = 16,                # Background strength higher is darker bg
        affinity_net_beta: int = 8,                     # Power strength

        # Segmentation
        semseg_name: str = 'name',
        semseg_batch_size = 4,
        semseg_dataset_root = '/',
        semseg_image_size = 513,
        semseg_epochs = 5,
        semseg_pretrained = False,
    ):
        self.sweep_id = sweep_id
        self.eval_dataset_root = eval_dataset_root

        self.classifier_dataset_root = classifier_dataset_root
        self.classifier_name = classifier_name
        self.classifier_epochs = classifier_epochs
        self.classifier_batch_size_train = classifier_batch_size_train
        self.classifier_image_size = classifier_image_size
        self.classifier_pretrained = classifier_pretrained
        self.classifier_pretrained_unfreeze = classifier_pretrained_unfreeze

        self.cams_save_gt_labels = cams_save_gt_labels
        self.cams_measure_batch_size = cams_measure_batch_size
        self.cams_produce_batch_size = cams_produce_batch_size
        self.cams_bg_alpha = cams_bg_alpha
        self.cams_bg_alpha_low = cams_bg_alpha_low
        self.cams_bg_alpha_high = cams_bg_alpha_high
        
        self.affinity_net_name = affinity_net_name
        self.affinity_net_image_size = affinity_net_image_size
        self.affinity_net_batch_size = affinity_net_batch_size
        self.affinity_net_epochs = affinity_net_epochs
        self.affinity_net_gamma = affinity_net_gamma
        self.affinity_net_log_t = affinity_net_log_t
        self.affinity_net_bg_alpha = affinity_net_bg_alpha
        self.affinity_net_beta = affinity_net_beta

        self.semseg_name = semseg_name
        self.semseg_batch_size = semseg_batch_size
        self.semseg_dataset_root = semseg_dataset_root
        self.semseg_image_size = semseg_image_size
        self.semseg_epochs = semseg_epochs
        self.semseg_pretrained = semseg_pretrained

    def toDictionary(self):
        return {
            'eval_dataset_root': self.eval_dataset_root,
            'classifier_dataset_root': self.classifier_dataset_root,
            'classifier_name': self.classifier_name,
            'classifier_epochs': self.classifier_epochs,
            'classifier_batch_size_train': self.classifier_batch_size_train,
            'classifier_image_size': self.classifier_image_size,
            'classifier_pretrained': self.classifier_pretrained,
            'classifier_pretrained_unfreeze': self.classifier_pretrained_unfreeze,

            'cams_save_gt_labels': self.cams_save_gt_labels,
            'cams_produce_batch_size': self.cams_produce_batch_size,
            'cams_measure_batch_size': self.cams_measure_batch_size,
            'cams_bg_alpha': self.cams_bg_alpha,
            'cams_bg_alpha_low': self.cams_bg_alpha_low,
            'cams_bg_alpha_high': self.cams_bg_alpha_high,

            'affinity_net_name': self.affinity_net_name,
            'affinity_net_image_size': self.affinity_net_image_size,
            'affinity_net_batch_size': self.affinity_net_batch_size,
            'affinity_net_epochs': self.affinity_net_epochs,
            'affinity_net_gamma': self.affinity_net_gamma,
            'affinity_net_log_t': self.affinity_net_log_t,
            'affinity_net_bg_alpha': self.affinity_net_bg_alpha,
            'affinity_net_beta': self.affinity_net_beta,

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