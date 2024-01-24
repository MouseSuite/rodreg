import nibabel as nib
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

def build_dict_rois(xmlfile):
    # Read XML for brain atlas
    xml_root = ET.parse(xmlfile).getroot()

    roi_names = list()
    roi_ids = list()

    for i in range(len(xml_root)):
        roi_names.append(xml_root[i].get('fullname'))
        roi_ids.append(int(xml_root[i].get('id')))

    roi_dict = {roi_ids[i]: roi_names[i]
                for i in range(len(roi_ids))}

    return roi_dict


def read_labels_file(labels_file):
    with open(labels_file, 'r') as file:
        labels = [line.strip().split() for line in file.readlines()]
    return {int(label[0]): label[1] for label in labels}

def calculate_roi_volumes(label_file, labels_dict):
    img = nib.load(label_file)
    voxel_volume_mm3 = abs(np.linalg.det(img.affine))
    data = img.get_fdata()
    
    roi_volumes = {}
    for label_num, description in labels_dict.items():
        roi_mask = (data == label_num)
        roi_volume_mm3 = voxel_volume_mm3 * roi_mask.sum()
        roi_volumes[label_num] = roi_volume_mm3
    
    return roi_volumes

def save_to_csv(output_file, roi_volumes):
    df = pd.DataFrame(list(roi_volumes.items()), columns=['ROI Number', 'ROI Volume (mm^3)'])
    df.to_csv(output_file, index=False)

# Example usage
nifti_label_file = '/deneb_disk/RodentTools/roiwise_stats_data/mouse_invivo_output/sub-12MomaleLW/anat/sub-12MomaleLW_T2w.label.nii.gz'
labels_xml_file = '/deneb_disk/RodentTools/data/MSA100/MSA100/MSA.xml'
output_csv_file = 'output_file.csv'

labels_dict = build_dict_rois(labels_xml_file)
roi_volumes = calculate_roi_volumes(nifti_label_file, labels_dict)
save_to_csv(output_csv_file, roi_volumes)
