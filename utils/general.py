import cv2
from cv2 import line
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model.pth')

def show_tranformed_image(train_loader, device, classes, colors):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    
    """
    if len(train_loader) > 0:
        for i in range(3):
            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            # Get all the predicited class names.
            pred_classes = [classes[i] for i in targets[i]['labels'].cpu().numpy()]
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            for box_num, box in enumerate(boxes):
                class_name = pred_classes[box_num]
                color = colors[classes.index(class_name)]
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            color, 2,
                            cv2.LINE_AA)
                cv2.putText(sample, classes[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, color, 2, cv2.LINE_AA)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenever called.

    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')

def save_loss_plot(OUT_DIR, train_loss_list, val_loss_list):
    """
    Function to save both train and validation loss graphs.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    :param val_loss_list: List containing the validation loss values.
    """
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss_list, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def save_train_loss_plot(OUT_DIR, train_loss_list):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def visualize_mosaic_images(boxes, labels, image_resized, classes):
    print(boxes)
    print(labels)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    for j, box in enumerate(boxes):
        color = (0, 255, 0)
        classn = labels[j]
        cv2.rectangle(image_resized,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
        cv2.putText(image_resized, classes[classn], 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
    cv2.imshow('Mosaic', image_resized)
    cv2.waitKey(0)

def save_model_state(epoch, model, optimizer, OUT_DIR):
    """
    Function to save the trained model till current epoch, or whenever called.
    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{OUT_DIR}/last_model_state.pth')

def save_train_loss_plot(OUT_DIR, train_loss_list):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def denormalize(x, mean=None, std=None):
    # 3, H, W, B
    # print(x.shape)
    # ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(x, 0, 1)

def save_validation_results(images, detections, counter, out_dir, classes, colors):
    """
    Function to save validation results.
    :param images: All the images from the current batch.
    :param detections: All the detection results.
    :param counter: Step counter for saving with unique ID.
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        # image_c = denormalize(image_c, IMG_MEAN, IMG_STD)
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))

        image = np.ascontiguousarray(image, dtype=np.float32)

        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        bboxes = detection['boxes'].detach().cpu().numpy()
        boxes = bboxes[scores >= 0.5].astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in labels.cpu().numpy()]
        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2, lineType=cv2.LINE_AA
            )
            cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/image_{i}_{counter}.jpg", image*255.)

def set_infer_dir():
    """
    This functions counts the number of inference directories already present
    and creates a new one in `outputs/inference/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/inference'):
        os.makedirs('outputs/inference')
    num_infer_dirs_present = len(os.listdir('outputs/inference/'))
    next_dir_num = num_infer_dirs_present + 1
    new_dir_name = f"outputs/inference/res_{next_dir_num}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name

def set_training_dir(dir_name=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
