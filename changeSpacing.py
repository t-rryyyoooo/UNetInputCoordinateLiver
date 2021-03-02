import argparse
import SimpleITK as sitk
from utils.imageProcessing.resample import resampleSpacing
from pathlib import Path

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00/imaging.nii.gz")
    parser.add_argument("save_path", help="$HOME/Desktop/data/kits19/case_00/imaging_resampled.nii.gz")
    parser.add_argument("--is_label", action="store_true")
    parser.add_argument("--spacing", default=[0.78, 0.78, 3.0], type=float, nargs=3)

    args = parser.parse_args()
    return args

def main(args):
    image = sitk.ReadImage(args.image_path)
    
    resampled_image = resampleSpacing(image, list(args.spacing), args.is_label)
    
    print("Change shape from {} to {}.".format(image.GetSize(), resampled_image.GetSize()))
    print("Chage spacing from {} to {}.".format(image.GetSpacing(), resampled_image.GetSpacing()))

    createParentPath(args.save_path)
    save_path = Path(args.save_path)
    save_path.parents.mkdir(parent=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(save_path), True)

if __name__=="__main__":
    args = parseArgs()
    main(args)
