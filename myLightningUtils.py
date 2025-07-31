
import torch
import torchio as tio
from lightning.pytorch.callbacks import BasePredictionWriter
import SimpleITK as sitk
import os
from torch.utils.data import DataLoader
import lightning as L
import json


def prepare_batch(batch):
    """Prepare input tensors and target tensors from the batch for model training"""
    # Concatenate all input channels (CT, masks, structures) into a single tensor
    inputs = torch.cat([
        batch['ct'][tio.DATA],
        batch['hrctv'][tio.DATA],
        batch['dwellpos_mask'][tio.DATA],
        batch['bladder'][tio.DATA],
        batch['rectum'][tio.DATA],
        batch['sigmoid'][tio.DATA],
        batch['hrctv_vag'][tio.DATA],
        batch['tandem'][tio.DATA],
        batch['ovoid'][tio.DATA],
        batch['ring'][tio.DATA],
        batch['needle'][tio.DATA]
        ],dim=1).float()
    # Extract dose as target
    targets = batch['dose'][tio.DATA].float()

    return inputs, targets


class CustomWriter(BasePredictionWriter):
    """Custom prediction writer to save model predictions as NIfTI files"""
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # Rescale prediction back to original dose range (0-100%)
        prediction = prediction.cpu().detach().squeeze().numpy()*10 # rescale back to 100%
        patient_id = batch['patient_id'][0]

        # Create output directory for this patient
        patient_output_dir = os.path.join(self.output_dir,patient_id)

        # Save the predicted dose
        sitk_img = sitk.GetImageFromArray(prediction)
        if not os.path.exists(patient_output_dir):
            os.makedirs(patient_output_dir)
        sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,"prediction.nii.gz"))

        # Save the ground truth dose
        true_dose = batch['dose'][tio.DATA].cpu().detach().squeeze().numpy()
        sitk_img = sitk.GetImageFromArray(true_dose)
        sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,"true_dose.nii.gz"))

        # Save all input channels as separate NIfTI files
        for key, value in batch.items():
            if key in ["metadata","patient_id"]:
                continue
            else:
                sitk_img = sitk.GetImageFromArray(value[tio.DATA].cpu().detach().squeeze().numpy())
                sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,key+".nii.gz"))












class BrachyDataModule(L.LightningDataModule):
    """Lightning DataModule for brachytherapy data handling"""
    def __init__(self, data_dir, batch_size=1, num_workers=0):
        super().__init__()
        # Set up data directories for train/val/test splits
        self.data_dir_train = os.path.join(data_dir,"Train")
        self.data_dir_val = os.path.join(data_dir,"Validation")
        self.data_dir_test = os.path.join(data_dir,"Test")
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        """Set up datasets for different stages (fit, validate, test, predict)"""
        if stage == 'fit' or stage is None:
            self.train_dataset = get_dataset(self.data_dir_train, training=True)
            self.val_dataset = get_dataset(self.data_dir_val)
        if stage == 'validate':
            self.val_dataset = get_dataset(self.data_dir_val)
        if stage == 'test':
            self.test_dataset = get_dataset(self.data_dir_test, test=True)

        if stage == 'predict':
            self.test_dataset = get_dataset(self.data_dir_test, test=True)


    def train_dataloader(self):
        """Return the training dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    def predict_dataloader(self):
        """Return the prediction dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Custom method to transfer batch data to the appropriate device"""
        if isinstance(batch, dict): # if it is a custom dataset
            for key, value in batch.items():
                # Skip metadata and patient ID (non-tensor data)
                if key in ["metadata","patient_id"] or "dp_" in key:
                    continue
                if type(value) == dict:
                    value[tio.DATA] = value[tio.DATA].to(device)
        return batch


def get_dataset(data_dir, training=False, test=False):
    """Create a TorchIO dataset from directory of patient data"""
    # Load appropriate subject list based on dataset type
    if test:
        subjects = get_subject_list_test(data_dir)
    else:
        subjects = get_subject_list(data_dir)

    # Define transformations - more augmentations for training data
    if training:
        transforms = tio.Compose([
                    tio.Resample("hrctv"),  # Resample to match HRCTV resolution
                    tio.transforms.Clamp(p=1.0, include=["ct"], out_min=-1000, out_max=1500),  # Clamp CT values to typical range
                    tio.transforms.RescaleIntensity(p=1.0, include=["ct"], in_min_max=(-1000, 1500),  out_min_max = (0,1)),  # Normalize CT values

                    # Data augmentation for training
                    tio.RandomFlip(p=0.8,axes=(0,1,2)),
                    tio.RandomAffine(scales=0.0, degrees=45, isotropic=True, translation=10.,p=0.8),
                    tio.CropOrPad(target_shape=(128,128,128)),  # Ensure consistent dimensions

                    ])
    else:
        transforms = tio.Compose([
                    tio.Resample("hrctv"),  # Resample to match HRCTV resolution
                    # No augmentation for validation/testing
                    tio.transforms.Clamp(p=1.0, include=["ct"], out_min=-1000, out_max=1500),
                    tio.transforms.RescaleIntensity(p=1.0, include=["ct"], in_min_max=(-1000, 1500),  out_min_max = (0,1)),
                    tio.CropOrPad(target_shape=(128,128,128)),  # Ensure consistent dimensions
                    ])

    # Create and return the dataset
    dataset = tio.SubjectsDataset(subjects, transform=transforms, load_getitem=False)
    return dataset

def get_subject_list(patients_dir, applicator = None, training=False):
    """Load patient subjects from directory for training/validation"""
    # List of patients to exclude
    banned = [

              ]
    patient_list = []

    for patient_folder in os.listdir(patients_dir):
        try:
            if patient_folder in banned:
                continue
            subject_dict = {}
            patient_dir = os.path.join(patients_dir,patient_folder)

            # Set patient ID
            subject_dict["patient_id"] = patient_folder

            # Load dose (ground truth)
            subject_dict["dose"] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs","dose.nii.gz"))

            # Load scalar images (applicator components and CT)
            for scalar_name in ["tandem", "ovoid","ring","needle","ct"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz")):
                    subject_dict[scalar_name] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz"))
                else:
                    raise Exception("Missing scalar: {}".format(scalar_name))

            # Load anatomical masks and structures
            for mask_name in ["hrctv","bladder","rectum","sigmoid","dwellpos_mask","hrctv_vag"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz")):
                    subject_dict[mask_name] = tio.LabelMap(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz"))
                else:
                    raise Exception("Missing mask: {}".format(mask_name))

            # Create subject and add to list
            subject = tio.Subject(subject_dict)
            patient_list.append(subject)
        except Exception as e:
            print("Error loading patient: {}, {}".format(patient_folder, e))
            continue
    return patient_list

def get_subject_list_test(patients_dir):
    """Load patient subjects from directory for testing/prediction, including metadata"""
    patient_list = []
    for patient_folder in os.listdir(patients_dir):
        try:
            subject_dict = {}
            patient_dir = os.path.join(patients_dir,patient_folder)
            subject_dict["patient_id"] = patient_folder

            # Load metadata for test patients
            metadata_file = os.path.join(patient_dir,'metadata',"metadata.json")
            with open(metadata_file) as f:
                metadata = json.load(f)
            subject_dict["metadata"] = metadata

            # Load dose (ground truth)
            subject_dict["dose"] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs","dose.nii.gz"))

            # Load scalar images (applicator components and CT)
            for scalar_name in ["tandem", "ovoid","ring","needle","ct"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz")):
                    subject_dict[scalar_name] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz"))
                else:
                    raise Exception("Missing scalar: {}".format(scalar_name))

            # Load anatomical masks and structures
            for mask_name in ["hrctv","bladder","rectum","sigmoid","dwellpos_mask","hrctv_vag"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz")):
                    subject_dict[mask_name] = tio.LabelMap(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz"))
                else:
                    raise Exception("Missing mask: {}".format(mask_name))

            # Load any dwell position files (prefixed with dp_)
            dp_list = [x for x in os.listdir(os.path.join(patient_dir,"NifTIs")) if "dp_" in x]
            for dp_name in dp_list:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",dp_name)):
                    subject_dict[dp_name.replace(".nii.gz","")] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",dp_name))
                else:
                    raise Exception("Missing scalar: {}".format(dp_name))

            # Create subject and add to list
            subject = tio.Subject(subject_dict)
            patient_list.append(subject)
        except Exception as e:
            print("Error loading patient: {}, {}".format(patient_folder, e))
            continue
    return patient_list



def prepare_batch(batch):
    inputs = torch.cat([
        batch['ct'][tio.DATA],
        batch['hrctv'][tio.DATA],
        batch['dwellpos_mask'][tio.DATA],
        batch['bladder'][tio.DATA],
        batch['rectum'][tio.DATA],
        batch['sigmoid'][tio.DATA],
        batch['hrctv_vag'][tio.DATA],
        batch['tandem'][tio.DATA],
        batch['ovoid'][tio.DATA],
        batch['ring'][tio.DATA],
        batch['needle'][tio.DATA]
        ],dim=1).float()
    targets = batch['dose'][tio.DATA].float()

    return inputs, targets


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        prediction = prediction.cpu().detach().squeeze().numpy()*10 # rescale back to 100%
        patient_id = batch['patient_id'][0]

        patient_output_dir = os.path.join(self.output_dir,patient_id)

        sitk_img = sitk.GetImageFromArray(prediction)
        if not os.path.exists(patient_output_dir):
            os.makedirs(patient_output_dir)
        sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,"prediction.nii.gz"))
        true_dose = batch['dose'][tio.DATA].cpu().detach().squeeze().numpy()
        sitk_img = sitk.GetImageFromArray(true_dose)
        sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,"true_dose.nii.gz"))

        for key, value in batch.items():
            if key in ["metadata","patient_id"]:
                continue
            else:
                sitk_img = sitk.GetImageFromArray(value[tio.DATA].cpu().detach().squeeze().numpy())
                sitk.WriteImage(sitk_img,os.path.join(patient_output_dir,key+".nii.gz"))












class BrachyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=1, num_workers=0):
        super().__init__()
        self.data_dir_train = os.path.join(data_dir,"Train")
        self.data_dir_val = os.path.join(data_dir,"Validation")
        self.data_dir_test = os.path.join(data_dir,"Test")
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = get_dataset(self.data_dir_train, training=True)
            self.val_dataset = get_dataset(self.data_dir_val)
        if stage == 'validate':
            self.val_dataset = get_dataset(self.data_dir_val)
        if stage == 'test':
            self.test_dataset = get_dataset(self.data_dir_test, test=True)

        if stage == 'predict':
            self.test_dataset = get_dataset(self.data_dir_test, test=True)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict): # if it is a custom dataset
            for key, value in batch.items():
                if key in ["metadata","patient_id"] or "dp_" in key:
                    continue
                if type(value) == dict:
                    value[tio.DATA] = value[tio.DATA].to(device)
        return batch


def get_dataset(data_dir, training=False, test=False):
    if test:
        subjects = get_subject_list_test(data_dir)
    else:
        subjects = get_subject_list(data_dir)
    if training:
        transforms = tio.Compose([
                    tio.Resample("hrctv"),
                    tio.transforms.Clamp(p=1.0, include=["ct"], out_min=-1000, out_max=1500),
                    tio.transforms.RescaleIntensity(p=1.0, include=["ct"], in_min_max=(-1000, 1500),  out_min_max = (0,1)), # https://arxiv.org/pdf/1809.10486.pdf


                    tio.RandomFlip(p=0.8,axes=(0,1,2)),
                    tio.RandomAffine(scales=0.0, degrees=45, isotropic=True, translation=10.,p=0.8),
                    tio.CropOrPad(target_shape=(128,128,128)),

                    ])
    else:
        transforms = tio.Compose([
                    tio.Resample("hrctv"),

                    tio.transforms.Clamp(p=1.0, include=["ct"], out_min=-1000, out_max=1500),
                    tio.transforms.RescaleIntensity(p=1.0, include=["ct"], in_min_max=(-1000, 1500),  out_min_max = (0,1)), # https://arxiv.org/pdf/1809.10486.pdf
                    tio.CropOrPad(target_shape=(128,128,128)),
                    ])
    dataset = tio.SubjectsDataset(subjects, transform=transforms, load_getitem=False)
    return dataset

def get_subject_list(patients_dir, applicator = None, training=False):
    banned = [

              ]
    patient_list = []

    for patient_folder in os.listdir(patients_dir):
        try:
            if patient_folder in banned:
                continue
            subject_dict = {}
            patient_dir = os.path.join(patients_dir,patient_folder)


            subject_dict["patient_id"] = patient_folder

            subject_dict["dose"] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs","dose.nii.gz"))
            for scalar_name in ["tandem", "ovoid","ring","needle","ct"]:
            # for scalar_name in ["unweighted_kernel"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz")):
                    subject_dict[scalar_name] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz"))
                else:
                    raise Exception("Missing scalar: {}".format(scalar_name))
            for mask_name in ["hrctv","bladder","rectum","sigmoid","dwellpos_mask","hrctv_vag"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz")):
                    subject_dict[mask_name] = tio.LabelMap(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz"))
                else:
                    raise Exception("Missing mask: {}".format(mask_name))

            subject = tio.Subject(subject_dict)
            patient_list.append(subject)
        except Exception as e:
            print("Error loading patient: {}, {}".format(patient_folder, e))

            continue
    return patient_list

def get_subject_list_test(patients_dir):
    patient_list = []
    for patient_folder in os.listdir(patients_dir):
        try:
            subject_dict = {}
            patient_dir = os.path.join(patients_dir,patient_folder)
            subject_dict["patient_id"] = patient_folder
            metadata_file = os.path.join(patient_dir,'metadata',"metadata.json")
            with open(metadata_file) as f:
                metadata = json.load(f)
            subject_dict["metadata"] = metadata
            subject_dict["dose"] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs","dose.nii.gz"))
            for scalar_name in ["tandem", "ovoid","ring","needle","ct"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz")):
                    subject_dict[scalar_name] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",scalar_name+".nii.gz"))
                else:
                    raise Exception("Missing scalar: {}".format(scalar_name))
            for mask_name in ["hrctv","bladder","rectum","sigmoid","dwellpos_mask","hrctv_vag"]:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz")):
                    subject_dict[mask_name] = tio.LabelMap(os.path.join(patient_dir,"NifTIs",mask_name+".nii.gz"))
                else:
                    raise Exception("Missing mask: {}".format(mask_name))

            dp_list = [x for x in os.listdir(os.path.join(patient_dir,"NifTIs")) if "dp_" in x]
            for dp_name in dp_list:
                if os.path.exists(os.path.join(patient_dir,"NifTIs",dp_name)):
                    subject_dict[dp_name.replace(".nii.gz","")] = tio.ScalarImage(os.path.join(patient_dir,"NifTIs",dp_name))
                else:
                    raise Exception("Missing scalar: {}".format(dp_name))
            subject = tio.Subject(subject_dict)
            patient_list.append(subject)
        except Exception as e:
            print("Error loading patient: {}, {}".format(patient_folder, e))
            continue
    return patient_list