import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from imageAndCoordinateExtractor import ImageAndCoordinateExtractor
from centerOfGravityCaluculater import CenterOfGravityCaluculater
from functions import getImageWithMeta, getSizeFromString

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/kits19/case_00000/segmentation.nii.gz")
    parser.add_argument("liver_path", help="$HOME/Desktop/data/kits19/case_00000/liver.mha")
    parser.add_argument("save_path", help="$HOME/Desktop/data/slice/hist_0.0/case_00000", default=None)
    parser.add_argument("patient_id", help="For making save_path")
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/label.mha")
    parser.add_argument("--image_patch_size", help="116-132-132", default="116-132-132")
    parser.add_argument("--label_patch_size", help="28-44-44", default="28-44-44")
    parser.add_argument("--with_nonmask", action="store_true")
    parser.add_argument("--overlap", help="1", type=int, default=1)
    parser.add_argument("--num_class", help="14", type=int, default=14)
    parser.add_argument("--class_axis", help="0", type=int, default=0)

    args = parser.parse_args()
    return args

def main(args):
    """ Read image and label. """
    label = sitk.ReadImage(args.label_path)
    image = sitk.ReadImage(args.image_path)
    liver = sitk.ReadImage(args.liver_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Get the patch size from string."""
    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)

    cogc = CenterOfGravityCaluculater(liver)
    liver_center = cogc.execute()

    print("Liver center", liver_center)
    iace = ImageAndCoordinateExtractor(
            image = image, 
            label = label,
            center = liver_center,
            mask = mask,
            image_array_patch_size = image_patch_size,
            label_array_patch_size = label_patch_size,
            overlap = args.overlap,
            num_class = args.num_class,
            class_axis = args.class_axis
            )

    iace.save(args.save_path, args.patient_id, with_nonmask=args.with_nonmask)
    """
    # For testing iace.outputRestoredImage.
    from tqdm import tqdm
    with tqdm(total=iace.__len__(), ncols=60, desc="Segmenting and restoreing...") as pbar:
        for ipa, lpa, cpa, mpa, _, index in iace.generateData():
            if (mpa > 0).any():
                lpa_onehot = np.eye(args.num_class)[lpa].transpose(3, 0, 1, 2)
                iace.insertToPredictedArray(index, lpa_onehot)

            pbar.update(1)

    predicted = iace.outputRestoredImage()
    pa = sitk.GetArrayFromImage(predicted)
    la = sitk.GetArrayFromImage(label)
    from functions import DICE
    dice = DICE(la, pa)
    print(dice)
    """

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
