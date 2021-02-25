import sys
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from coordinateArrayCreater import CoordinateArrayCreater
from itertools import product
from functions import paddingForNumpy, croppingForNumpy, clippingForNumpy, caluculatePaddingSize, getImageWithMeta, createParentPath
from utils import PatchGenerator

class ImageAndCoordinateExtractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    Mainly numpy!
    
    """
    def __init__(self, image, label, center=(0, 0, 0), mask=None, image_array_patch_size=[16, 48, 48], label_array_patch_size=[16, 48, 48], overlap=1, num_class=14, class_axis=0):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clippingForNumpy, shit the clip position by slide

        """
        
        self.org = image
        self.image_array = sitk.GetArrayFromImage(image)
        self.label_array = sitk.GetArrayFromImage(label)

        """ Implementing np.ones_like means the whole area is masked. """
        if mask is not None:
            self.mask_array = sitk.GetArrayFromImage(mask)
        else:
            self.mask_array = np.ones_like(self.image_array)

        self.center = center

        """ patch_size = [z, y, x] """
        self.image_array_patch_size = np.array(image_array_patch_size)
        self.label_array_patch_size = np.array(label_array_patch_size)

        self.overlap = overlap
        self.slide = self.label_array_patch_size // overlap

        self.makeGenerator()

        """ After implemeting makeGenerator(), self.label is padded to clip correctly. """
        self.num_class = num_class
        self.class_axis = class_axis
        self.predicted_array = np.zeros([num_class] + list(self.label_array.shape))
        self.counter_array = np.zeros_like(self.label_array)

    def makeGenerator(self):
        """ Caluculate paddingForNumpy size for label and image to clip correctly. """
        self.lower_pad_size, self.upper_pad_size = caluculatePaddingSize(
                                                    np.array(self.label_array.shape), 
                                                    self.image_array_patch_size, 
                                                    self.label_array_patch_size, self.slide
                                                    )

        """ Pad image and label. """
        self.image_array = paddingForNumpy(
                            self.image_array, 
                            self.lower_pad_size[0].tolist(), 
                            self.upper_pad_size[0].tolist()
                            )
        self.label_array = paddingForNumpy(
                            self.label_array, 
                            self.lower_pad_size[1].tolist(), 
                            self.upper_pad_size[1].tolist()
                            )

        self.mask_array = paddingForNumpy(
                            self.mask_array,
                            self.lower_pad_size[1].tolist(), 
                            self.upper_pad_size[1].tolist()
                            )

        """ If self.center is not None, get coordinate array. """
        cac = CoordinateArrayCreater(
                image_array = self.image_array,
                center = self.center
                )

        cac.execute()
        coordinate_array = cac.getCoordinate(kind="relative")

        """ Make generator for image, label, mask and coordinate. """
        self.image_patch_array_generator = PatchGenerator(
                                    self.image_array, 
                                    self.image_array_patch_size,
                                    self.slide
                                )
        self.label_patch_array_generator = PatchGenerator(
                                    self.label_array,
                                    self.label_array_patch_size,
                                    self.slide
                                )
        self.mask_patch_array_generator = PatchGenerator(
                                self.mask_array,
                                self.label_array_patch_size,
                                self.slide
                                )

        """ Make patch size and slide for 4 dimension because coordinate array has 4 dimention. """
        ndim = self.image_array.ndim
        coord_array_patch_size = np.array([ndim] + self.image_array_patch_size.tolist())
        coord_slide = np.array([ndim] + self.slide.tolist())

        self.coord_patch_array_generator = PatchGenerator(
                                    coordinate_array,
                                    coord_array_patch_size,
                                    coord_slide
                                    )


    def __len__(self):
        return self.image_patch_array_generator.__len__()
    def generateData(self):
        """ [1] means patch array because PatchGenerator returns index and patch_array. """
        for ipa, lpa, cpa, mpa in zip(self.image_patch_array_generator(), self.label_patch_array_generator(), self.coord_patch_array_generator(), self.mask_patch_array_generator()):

            input_index = ipa[0]
            output_index = lpa[0]
            yield ipa[1], lpa[1], cpa[1], mpa[1], input_index, output_index

    def save(self, save_path, patient_id, with_nonmask=False):
        if not isinstance(patient_id, str):
            patient_id = str(patient_id)

        save_path = Path(save_path)

        save_mask_path = save_path / "mask" / "case_{}".format(patient_id.zfill(2))
        save_mask_path.mkdir(parents=True, exist_ok=True)

        if with_nonmask:
            save_nonmask_path = save_path / "nonmask" / "case_{}".format(patient_id.zfill(2))
            save_nonmask_path.mkdir(parents=True, exist_ok=True)

        if with_nonmask:
            desc = "Saving masked and nonmasked images, labels and coordinates..."
        else:
            desc = "Saving masked images, labels and coordinates..."

        with tqdm(total=self.image_patch_array_generator.__len__(), ncols=100, desc=desc) as pbar:
            for i, (ipa, lpa, cpa, mpa, _, _) in enumerate(self.generateData()):
                if (mpa > 0).any():
                    save_masked_image_path = save_mask_path / "image_{:04d}.npy".format(i)
                    save_masked_label_path = save_mask_path / "label_{:04d}.npy".format(i)
                    save_masked_coord_path = save_mask_path / "coord_{:04d}.npy".format(i)

                    np.save(str(save_masked_image_path), ipa)
                    np.save(str(save_masked_label_path), lpa)
                    np.save(str(save_masked_coord_path), cpa)

                else:
                    if with_nonmask:
                        save_nonmasked_image_path = save_nonmask_path / "image_{:04d}.npy".format(i)
                        save_nonmasked_label_path = save_nonmask_path / "label_{:04d}.npy".format(i)
                        save_nonmasked_coord_path = save_nonmask_path / "coord_{:04d}.npy".format(i)
             
                        np.save(str(save_nonmasked_image_path), ipa)
                        np.save(str(save_nonmasked_label_path), lpa)
                        np.save(str(save_nonmasked_coord_path), cpa)

                pbar.update(1)

           
    def insertToPredictedArray(self, index, array):
        """ Insert predicted array (before argmax array) which has probability per class. """

        assert array.ndim == self.predicted_array.ndim

        predicted_slices = []
        counter_slices = []
        s = slice(0, self.num_class)
        predicted_slices.append(s)
        for i in range(self.label_array.ndim):
            s = slice(index[i], index[i] + self.label_array_patch_size[i])
            predicted_slices.append(s)
            counter_slices.append(s)

        predicted_slices = tuple(predicted_slices)
        counter_slices = tuple(counter_slices)
        

        """ Array's shape and counter's array shape are not same, which leads to shape error, so, address it. """
        s = np.delete(np.arange(array.ndim), self.class_axis)
        s = np.array(array.shape)[s]

        self.predicted_array[predicted_slices] = array
        self.counter_array[counter_slices] = np.ones(s)

    def outputRestoredImage(self):
        """ Usually, this method is used after all of predicted patch array is insert to self.predicted_array with insertToPredictedArray. """

        """ Address division by zero. """
        self.counter_array = np.where(self.counter_array == 0, 1, self.counter_array)

        self.predicted_array /= self.counter_array
        self.predicted_array = np.argmax(self.predicted_array, axis=self.class_axis)
        self.predicted_array = croppingForNumpy(
                                self.predicted_array, 
                                self.lower_pad_size[1].tolist(),
                                self.upper_pad_size[1].tolist()
                                )

        predicted = getImageWithMeta(self.predicted_array, self.org)

        return predicted
