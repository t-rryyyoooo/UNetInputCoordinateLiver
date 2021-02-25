import SimpleITK as sitk
import sys
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta, getSizeFromString
from pathlib import Path
from imageAndCoordinateExtractor import ImageAndCoordinateExtractor
from centerOfGravityCaluculater import CenterOfGravityCaluculater
from tqdm import tqdm
import torch
import cloudpickle
import re


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("liver_path", help="$HOME/Desktop/data/kits19/case_00000/lver.nii.gz")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/mask.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="44-44-28")
    parser.add_argument("--label_patch_size", help="44-44-28", default="44-44-28")
    parser.add_argument("--overlap", help="1", default=1, type=int)
    parser.add_argument("--num_class", help="14", default=14, type=int)
    parser.add_argument("--class_axis", help="0", default=0, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    sys.path.append("..")
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Slice module. """

    image = sitk.ReadImage(args.image_path)
    liver = sitk.ReadImage(args.liver_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Dummy image """
    label = sitk.Image(image.GetSize(), sitk.sitkInt8)
    label.SetOrigin(image.GetOrigin())
    label.SetDirection(image.GetDirection())
    label.SetSpacing(image.GetSpacing())

    """ Get the patch size from string."""
    image_patch_size = getSizeFromString(args.image_patch_size)

    """ Get the patch size from string."""
    label_patch_size = getSizeFromString(args.label_patch_size)

    cogc = CenterOfGravityCaluculater(liver)
    liver_center = cogc.execute()

    print("Liver center: ", liver_center)
    
    iace = ImageAndCoordinateExtractor(
            image = image, 
            label = label, 
            mask = mask,
            image_array_patch_size = image_patch_size, 
            label_array_patch_size = label_patch_size, 
            overlap = args.overlap, 
            center = liver_center,
            num_class = args.num_class,
            class_axis = args.class_axis
            )

    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    segmented_array_list = []
    with tqdm(total=iace.__len__(), ncols=60, desc="Segmenting and restoring...") as pbar:
        for image_patch_array, lpa, coord_patch_array, mask_patch_array, _, index in iace.generateData():
            if (mask_patch_array > 0).any():
                while image_patch_array.ndim < 5:
                    image_patch_array = image_patch_array[np.newaxis, ...]

                while coord_patch_array.ndim < 5:
                    coord_patch_array = coord_patch_array[np.newaxis, ...]

                image_patch_array = torch.from_numpy(image_patch_array).to(device, dtype=torch.float)
                coord_patch_array = torch.from_numpy(coord_patch_array).to(device, dtype=torch.float)
                
                segmented_array = model(image_patch_array, coord_patch_array).to("cpu").detach().numpy().astype(np.float)
                segmented_array = np.squeeze(segmented_array)

                iace.insertToPredictedArray(index, segmented_array)

            pbar.update(1)

    segmented = iace.outputRestoredImage()

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
