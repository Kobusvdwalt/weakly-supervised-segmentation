class Config:
    # Experiment erase
    KEY_ERASE_SWEEP_ID = 'ERASE_SWEEP_ID'
    KEY_ERASE_TRAIN_DATASET = 'ERASE_TRAIN_DATASET'
    KEY_ERASE_EVAL_DATASET = 'ERASE_EVAL_DATASET'
    KEY_ERASE_MODEL = 'ERASE_MODEL'
    KEY_ERASE_MODE = 'ERASE_MODE'
    KEY_ERASE_BATCH_SIZE = 'ERASE_BATCH_SIZE'
    KEY_ERASE_EPOCHS = 'ERASE_EPOCHS'


    # Experiment psa - classifier

    # Experiment psa - cams

    # Experiment psa - affnet

    # Experiment psa - segmentation
    
    def __init__(self, config):
        self.config = config

    def toDictionary(self):
        return self.config

    def getValue(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise Exception('Config key not found: ' + key)

class ConfigManager():
    def __init__(self):
        self.setConfig(Config({}))

    def getConfig(self) -> Config:
        return self.config

    def setConfig(self, config):
        self.config = config

config_manager = ConfigManager()


# class ConfigExpErase:
#     def __init__(self,
#         sweep_id: str = '0',
#         eval_dataset_root: str = '/',

#         # Classifier
#         classifier_dataset_root: str = '/',
#         classifier_name: str = 'name',
#         classifier_epochs: int = 1,
#         classifier_batch_size_train: int = 8,
#         classifier_image_size: int = 448,
#         classifier_pretrained: bool = True,
#         classifier_pretrained_unfreeze: int = 0,
#         classifier_erase_type: str = 'none',
#         classifier_erase_strength: str = 1,

#     ):
#         self.sweep_id = sweep_id
#         self.eval_dataset_root = eval_dataset_root

#         self.classifier_dataset_root = classifier_dataset_root
#         self.classifier_name = classifier_name
#         self.classifier_epochs = classifier_epochs
#         self.classifier_batch_size_train = classifier_batch_size_train
#         self.classifier_image_size = classifier_image_size
#         self.classifier_pretrained = classifier_pretrained
#         self.classifier_pretrained_unfreeze = classifier_pretrained_unfreeze
#         self.classifier_erase_type = 'none'
#         self.classifier_erase_strength = 1

#     def toDictionary(self):
#         return {
#             'eval_dataset_root': self.eval_dataset_root,
#             'classifier_dataset_root': self.classifier_dataset_root,
#             'classifier_name': self.classifier_name,
#             'classifier_epochs': self.classifier_epochs,
#             'classifier_batch_size_train': self.classifier_batch_size_train,
#             'classifier_image_size': self.classifier_image_size,
#             'classifier_pretrained': self.classifier_pretrained,
#             'classifier_pretrained_unfreeze': self.classifier_pretrained_unfreeze,
#             'classifier_erase_type': self.classifier_erase_type,
#             'classifier_erase_strength': self.classifier_erase_strength,
#         }




        # sweep_id: str = '0',
        # eval_dataset_root: str = '/',

        # # Classifier
        # classifier_dataset_root: str = '/',
        # classifier_name: str = 'name',
        # classifier_epochs: int = 1,
        # classifier_batch_size_train: int = 8,
        # classifier_image_size: int = 448,
        # classifier_pretrained: bool = False,
        # classifier_pretrained_unfreeze: int = 0,

        # # Cams
        # cams_produce_batch_size: int = 32,
        # cams_measure_batch_size: int = 64,
        # cams_save_gt_labels: bool = False,
        # cams_bg_alpha: int = 16,
        # cams_bg_alpha_low: int = 4,
        # cams_bg_alpha_high: int = 32,
        
        # # AffinityNet
        # affinity_net_name: str = 'affinitynet',
        # affinity_net_image_size: int = 448,
        # affinity_net_batch_size: int = 8,
        # affinity_net_epochs: int = 6,
        # affinity_net_gamma: int = 5,                    # Pixel range for affnet 
        # affinity_net_log_t: int = 8,                    # Matmul count
        # affinity_net_bg_alpha: int = 16,                # Background strength higher is darker bg
        # affinity_net_beta: int = 8,                     # Power strength

        # # Segmentation
        # semseg_name: str = 'deeplab',
        # semseg_batch_size = 4,
        # semseg_dataset_root = '/',
        # semseg_image_size = 513,
        # semseg_epochs = 5,
        # semseg_pretrained = False,