import sys
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from coordinateArrayCreater import CoordinateArrayCreater
from itertools import product
from functions import paddingForNumpy, croppingForNumpy, clippingForNumpy, caluculatePaddingSize, getImageWithMeta, createParentPath

class ImageAndCoordinateExtractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    Mainly numpy!
    
    """
    def __init__(self, image, label, center=(0, 0, 0), mask=None, image_array_patch_size=[16, 48, 48], label_array_patch_size=[16, 48, 48], overlap=1, integrate=False, stacking=False):
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
        if mask is not None:
            self.mask_array = sitk.GetArrayFromImage(mask)
        else:
            self.mask_array = None

        self.center = center

        """ patch_size = [z, y, x] """
        self.image_array_patch_size = np.array(image_array_patch_size)
        self.label_array_patch_size = np.array(label_array_patch_size)

        self.overlap = overlap
        self.slide = self.label_array_patch_size // overlap
        self.integrate = integrate
        self.stacking = stacking

    def execute(self):
        """ Clip image and label. """

        """ Caluculate each paddingForNumpy size for label and image to clip correctly. """
        self.lower_pad_size, self.upper_pad_size = caluculatePaddingSize(np.array(self.label_array.shape), self.image_array_patch_size, self.label_array_patch_size, self.slide)

        """ Pad image and label. """
        self.image_array = paddingForNumpy(self.image_array, self.lower_pad_size[0].tolist(), self.upper_pad_size[0].tolist())
        self.label_array = paddingForNumpy(self.label_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        if self.mask_array is not None:
            self.mask_array = paddingForNumpy(self.mask_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())


        """ If self.center is not None, get coordinate array. """
        cac = CoordinateArrayCreater(
                image_array = self.image_array,
                center = self.center
                )
        cac.execute()
        coordinate_array = cac.getCoordinate(kind="relative")

        """ Clip the image and label to patch size. """
        self.image_array_patch_list = self.makePatch(self.image_array, self.image_array_patch_size, self.slide, desc="images")
        self.label_array_patch_list = self.makePatch(self.label_array, self.label_array_patch_size, self.slide, desc="labels")
        if self.mask_array is not None:
            self.mask_array_patch_list = self.makePatch(self.mask_array, self.label_array_patch_size, self.slide, desc="masks")

        """ Create stacked image. """
        if self.stacking:
            print("Stacking images...")
            min_value = self.image_array.min()
            self.image_array_patch_list = self.createStackedImageArrayList(min_value)

        """ Make patch size and slide for 4 dimension because coordinate array has 4 dimention. """
        ndim = self.image_array.ndim
        coordinate_array_patch_size = np.array([ndim] + self.image_array_patch_size.tolist())
        coordinate_slide = np.array([ndim] + self.slide.tolist())

        self.coordinate_array_patch_list = self.makePatch(coordinate_array, coordinate_array_patch_size, coordinate_slide, desc="coordinates")

        
        """ Confirm if makePatch runs correctly. """
        assert len(self.image_array_patch_list) == len(self.label_array_patch_list) == len(self.coordinate_array_patch_list)

        if self.mask_array is not None:
            assert len(self.image_array_patch_list) == len(self.mask_array_patch_list)

        """ Check mask. """
        self.masked_indices = []
        self.nonmasked_indices = []
        with tqdm(len(self.image_array_patch_list), desc="Checking mask...", ncols=60) as pbar:
            for i in range(len(self.image_array_patch_list)):
                if self.mask_array is not None:
                    if (self.mask_array_patch_list[i] == 0).all():
                        self.nonmasked_indices.append(i)

                    else:
                        self.masked_indices.append(i)

                else:
                    self.masked_indices.append(i)

                pbar.update(1)


    def checkExistence(self, target, center, lengths):
        length = lengths[1] * lengths[2]
        max_length = lengths[0] * lengths[1] * lengths[2]
        num_z_layer = center // length
        num_y_layer = center // lengths[2]

        """ Target is in range or not. """
        if (target < 0) or (max_length <= target):
            return False

        # Target is the same layer in z axis.
        if target < (num_z_layer * length) or ((num_z_layer + 1) * length) <= target:
            return False

        # Left edge judgement.
        location = center % lengths[1]
        if location in [i for i in range(3)]:
            i = 3 - location

            lower_left = center - lengths[1]*3 - 3 
            left = center - 3 
            upper_left = center + lengths[1]*3 - 3 
            if target in [lower_left, left, upper_left]:
                return False

        # RIght edge judgement.
        if location in [lengths[1] - i for i in range(1, 4)]:
            i = 3 - (lengths[1] - location - 1)

            lower_right = center - lengths[1]*3 + 3
            right = center + 3
            upper_right = center + lengths[1]*3 + 3
            if target in [lower_right, right, upper_right]:
                return False

        return True
            
    def createIndicesForStacking(self, i, no_existence_number=-100):
        sizes = np.array(self.image_array.shape) - self.image_array_patch_size 
        lengths = [((size + 1) // slide) + 1 for size, slide in zip(sizes, self.slide)]
        l = lengths[2]
        #directions = [-l - 1, -l, -l + 1, -1, +1, +l - 1, +l, +l + 1]
        directions = [-l*3 - 3, -l*3, -l*3 + 3, -3, +3, +l*3 - 3, +l*3, +l*3 + 3]
        stacked_indices = []
        stacked_indices.append(i)
        for direction in directions:
            d = i + direction
            existence = self.checkExistence(d, i, lengths)
            if existence:
                stacked_indices.append(d)
            else:
                stacked_indices.append(no_existence_number)

        return stacked_indices


    def createStackedImageArrayList(self, min_value, no_existence_number=-100):
        min_array = np.zeros_like(self.image_array_patch_list[0]) + min_value

        stacked_image_array_list = []
        for i in range(len(self.image_array_patch_list)):
            stacked_image_array = []
            indices = self.createIndicesForStacking(i)
            for index in indices:
                if index == no_existence_number:
                    stacked_image_array.append(min_array)
                else:
                    stacked_image_array.append(self.image_array_patch_list[index])

            stacked_image_array = np.stack(stacked_image_array)

            stacked_image_array_list.append(stacked_image_array)

        return stacked_image_array_list

    def loadData(self, nonmask=False):
        if not self.integrate:
            if nonmask:
                for i in self.nonmasked_indices:
                    yield self.image_array_patch_list[i], self.label_array_patch_list[i], self.coordinate_array_patch_list[i]

            else:
                for i in self.masked_indices:
                    yield self.image_array_patch_list[i], self.label_array_patch_list[i], self.coordinate_array_patch_list[i]
        
        else:
            if nonmask:
                for i in self.nonmasked_indices:
                    image_array_patch = np.concatenate([self.image_array_patch_list[i][np.newaxis, ...], self.coordinate_array_patch_list[i]])
                    yield image_array_patch, self.label_array_patch_list[i]

            else:
                for i in self.masked_indices:
                    image_array_patch = np.concatenate([self.image_array_patch_list[i][np.newaxis, ...], self.coordinate_array_patch_list[i]])
                    yield image_array_patch, self.label_array_patch_list[i]


    def makePatch(self, image_array, patch_size, slide, desc):
        size = np.array(image_array.shape) - patch_size 
        indices = []
        for i in range(image_array.ndim):
            r = range(0, size[i] + 1, slide[i])
            indices.append(r)
        indices = [i for i in product(*indices)]

        patch_list = []
        with tqdm(total=len(indices), desc="Clipping {}...".format(desc), ncols=60) as pbar:
            for index in indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + patch_size

                patch = clippingForNumpy(image_array, lower_clip_size, upper_clip_size)
                patch_list.append(patch)

                pbar.update(1)

        return patch_list

    def save(self, save_path, nonmask=False):
        save_path = Path(save_path)
        save_image_path = save_path / "dummy.npy"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        if not self.integrate:
            for i, (image_array_patch, label_array_patch, coordinate_array_patch)  in tqdm(enumerate(self.loadData(nonmask=nonmask))):
                save_image_path = save_path / "image_{:04d}.npy".format(i)
                save_label_path = save_path / "label_{:04d}.npy".format(i)
                save_coordinate_path = save_path / "coordinate_{:04d}.npy".format(i)
                np.save(str(save_image_path), image_array_patch)
                np.save(str(save_label_path), label_array_patch)
                np.save(str(save_coordinate_path), coordinate_array_patch)

        else:
            for i, (image_array_patch, label_array_patch)  in tqdm(enumerate(self.loadData(nonmask=nonmask))):
                save_image_path = save_path / "image_{:04d}.npy".format(i)
                save_label_path = save_path / "label_{:04d}.npy".format(i)

                np.save(str(save_image_path), image_array_patch)
                np.save(str(save_label_path), label_array_patch)


    def restore(self, predict_array_list):
        predict_array = np.zeros_like(self.label_array)

        size = np.array(self.label_array.shape) - self.label_array_patch_size 

        indices = []
        for i in range(self.label_array.ndim):
            r = range(0, size[i] + 1, self.slide[i])
            indices.append(r)
        indices = np.array([i for i in product(*indices)])

        with tqdm(total=len(predict_array_list), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, idx in zip(predict_array_list, indices[self.masked_indices]): 
                slices = []
                for i in range(self.label_array.ndim):
                    s = slice(idx[i], idx[i] + self.label_array_patch_size[i])
                    slices.append(s)
                slices = tuple(slices)

                predict_array[slices] = pre_array
                pbar.update(1)


        predict_array = croppingForNumpy(predict_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        predict = getImageWithMeta(predict_array, self.org)
        predict.SetOrigin(self.org.GetOrigin())
        

        return predict






