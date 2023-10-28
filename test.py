import glob


c = 0
tot_classes = {}
for img_path in glob.glob('/workspaces/dog-breed-classification-webapp/dog-breeds-classification/dogImages/train/*/*'):
    # to get breed type and remove number out from breed name
    breed_type = img_path.split('/')[-2][4:]
    if breed_type not in tot_classes.keys():
        tot_classes[breed_type] = c
        c += 1

# convert the label back
# label2idx, idx2label
reverse_tot_classes = {v:k for k, v in tot_classes.items()}

print(tot_classes)
print(reverse_tot_classes)