import csv
import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options import TestOptions
from data import create_dataloader, create_dataloader_new


from data.process import get_processing_model

from data.datasets import loadpathslist,custom_augment,process_img

from PIL import Image






def validate_PSM(model, data_loader):
    print("Validate_PSM")
    y_true, y_pred = [], []
    i = 0
    with torch.no_grad():
        for data in data_loader:
            i += 1
            print("batch number {}/{}".format(i, len(data_loader)), end='\r')
            input_img = data[0]  # [batch_size, 3, height, width]
            cropped_img = data[1].to('cuda:1')  # [batch_size, 3, 224, 224]
            label = data[2].to('cuda:1')  # [batch_size, 1]
            scale = data[3].to('cuda:1')  # [batch_size, 1, 2]
            logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    return y_true, y_pred


def validate_single(model, opt):
    print("Validate_single")
    opt=get_processing_model(opt)
    real_img_list = loadpathslist(opt.dataroot,'0_real')
    real_label_list = [0 for _ in range(len(real_img_list))]
    fake_img_list = loadpathslist(opt.dataroot,'1_fake')
    fake_label_list = [1 for _ in range(len(fake_img_list))]
    imgs = real_img_list+fake_img_list
    labels = real_label_list+fake_label_list
    y_true, y_pred = [], []
    if opt.detect_method == "Fusing":
        data_loader = create_dataloader_new(opt)
        y_true, y_pred = validate_PSM(model, data_loader)
    else:
        # with torch.no_grad():

        for idx in range(len(imgs)):
            print("batch number {}/{}".format(idx, len(imgs)), end='\r')
            img = Image.open(imgs[idx]).convert('RGB')
            img = custom_augment(img, opt)
            img,target=process_img(img,opt,imgs[idx],labels[idx])
            in_tens = img.unsqueeze(0)
            in_tens = in_tens.to('cuda:1')
            # label = label.to('cuda:1')
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend([labels[idx]])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


def validate(model, opt):
    print("Validate")

    opt = get_processing_model(opt)

    data_loader = create_dataloader_new(opt)
    y_true, y_pred = [], []
    if opt.detect_method == "Fusing":
        y_true, y_pred = validate_PSM(model, data_loader)
    else:
        # with torch.no_grad():
        i = 0
        with open('output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header (optional)
            writer.writerow(['Name', 'Label', 'y_pred'])

            for img, label, name in data_loader:
                i += 1
                print("batch number {}/{}".format(i, len(data_loader)), end='\r')
                in_tens = img.to('cuda:1')
                # label = label.to('cuda:1')
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
                y_pred_class = [1 if prob > 0.5 else 0 for prob in y_pred]

                for n, l, p in zip(name, label, y_pred_class):
                    writer.writerow([n, l.item(), p])


    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to('cuda:1')
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
