#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name extractImageAndCoordinate.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file.
if [ $which = "y" ];then
 JSON_NAME="extractImageAndCoordinate.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

# From json file, read required variables.
readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly STACKING=$(cat ${JSON_FILE} | jq -r ".stacking")
readonly NONMASK=$(cat ${JSON_FILE} | jq -r ".nonmask")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly LOG_FILE=$(eval echo $(cat ${JSON_FILE} | jq -r ".log_file"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly LIVER_NAME=$(cat ${JSON_FILE} | jq -r ".liver_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")

echo "DATA_DIRECTORY:${DATA_DIRECTORY}"
echo "SAVE_DIRECTORY:${SAVE_DIRECTORY}"
echo "LOG_FILE:${LOG_FILE}"

# Make directory to save LOG.
mkdir -p `dirname ${LOG_FILE}`
date >> $LOG_FILE

for number in ${NUM_ARRAY[@]}
do
 data="${DATA_DIRECTORY}/case_${number}"
 image="${data}/${IMAGE_NAME}"
 label="${data}/${LABEL_NAME}"
 liver="${data}/${LIVER_NAME}"
 save="${SAVE_DIRECTORY}"

 echo "Image:${image}"
 echo "Label:${label}"
 echo "Liver:${liver}"
 echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
 echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
 echo "OVERLAP:${OVERLAP}"
 echo "STACKING:${STACKING}"

 if [ $MASK_NAME = "No" ];then
  echo "Mask:${MASK_PATH}"
  mask=""

 else
  mask_path="${data}/${MASK_NAME}"
  echo "Mask:${mask_path}"
  mask="--mask_path ${mask_path}"

  echo "NONMASK:${NONMASK}"
  if $NONMASK ;then
   nonmask="--nonmask"
   save="${save}/nonmask"

  else
   nonmask=""
   save="${save}/mask"

  fi
 fi

 if $STACKING ;then
  stacking="--stacking"
  save="${save}/stacking"

 else
  stacking=""
 fi

 save="${save}/image/case_${number}"
 echo "Save_path:${save}"

 python3 extractImageAndCoordinate.py ${image} ${label} ${liver} ${save} --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --overlap ${OVERLAP} ${stacking} ${mask} ${nonmask} 

 # Judge if it works.
 if [ $? -eq 0 ]; then
  echo "case_${number} done."
 
 else
  echo "case_${number}" >> $LOG_FILE
  echo "case_${number} failed"
 
 fi

done


