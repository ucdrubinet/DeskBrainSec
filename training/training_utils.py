import torch
import os
import time
import glob
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, hamming_loss, confusion_matrix

def train_step(model, optimizer, plaque_loader, tissue_loader, device,
                plaque_loss_fn, tissue_loss_fn, epoch, num_epochs, stat_count, scheduler):
    """
    Execute one epoch of training

    Args:
        model: initialized PlaqueTissueClassifier
        optimizer: initialized torch.optim
        plaque_loader: initialized DataLoader with PlaqueDataset
        tissue_loader: initialized DataLoader with WMGMDataset
        device: one of ['gpu', 'cpu', 'mps']
        plaque_loss_fn: multilabel loss function
        tissue_loss_fn: multiclass loss function
        epoch: current epoch number
        num_epochs total epochs in training
        stat_count: print stats every stat_count steps
        scheduler: LRScheduler or None
    
    Returns:
        Dictionary of keys "tissue" and "plaque" with lists of loss values each step
    """
    batch_losses = {"plaque": [], "tissue": []}
    # Set number of steps to fit larger dataset
    num_steps = max(len(plaque_loader), len(tissue_loader))
    # Get next batch from datasets
    for train_ct in range(num_steps):
        try:
            p_data = next(p_labelled_iter)
        except:
            p_labelled_iter = iter(plaque_loader)
            p_data = next(p_labelled_iter)
        try:
            t_data = next(t_labelled_iter)
        except:
            t_labelled_iter = iter(tissue_loader)
            t_data = next(t_labelled_iter)

        # Set to training mode
        with torch.enable_grad():
            model.train()
            # Move batch from the datasets to device
            t_images, t_labels = t_data[0].to(device,dtype=torch.float), t_data[1].to(device,dtype=torch.long)
            p_images, p_labels = p_data[0].to(device,dtype=torch.float), p_data[1].to(device,dtype=torch.long)
            # Reset optimizer
            optimizer.zero_grad()

            # Get plaque and tissue predictions for the batches
            p_preds, _ = model(p_images)
            _, t_preds = model(t_images)

            # Calculate plaque and tissue loss
            p_loss = plaque_loss_fn(p_preds, p_labels)
            t_loss = tissue_loss_fn(t_preds, t_labels)
            # Sum losses and update model weights
            loss = p_loss + t_loss
            loss.backward()

            # Record loss for each dataset
            batch_losses["plaque"].append(p_loss.item())
            batch_losses["tissue"].append(t_loss.item())

            # Update optimizer
            optimizer.step()
        
        # Print statistics on every stat_count iteration
        if (train_ct+1) % stat_count == 0:
            print('Epoch [%d/%d], Step [%d/%d], Plaque Loss: %.4f, Tissue Loss: %.4f'
                                    %(epoch+1, num_epochs, train_ct+1,
                                    num_steps, batch_losses["plaque"][-1],
                                      batch_losses["tissue"][-1]))
        
        if scheduler:
            scheduler.step()
    # Return loss at each step of epoch
    return batch_losses

def eval_step(model, plaque_loader, tissue_loader, device):
    """
    Perform inference on model

    Args:
        model: PlaqueTissueClassifier instance
        plaque_loader: DataLoader instance with PlaqueDataset
        tissue_loader: DataLoader instance with WMGMDataset
        device: one of ['cuda', 'cpu', 'mps']

    Returns:
        Predictions dictionary with plaque and tissue inferred labels lists
        Labels dictionary with plaque and tissue real labels lists
        Note: values are lists of numpy array labels
    """
    # Set to inference mode
    with torch.no_grad():
        model.eval()

        # init metrics
        predictions = {"plaque": [], "tissue": []}
        labels = {"plaque": [], "tissue": []}

        # Plaque inference
        for p_data in plaque_loader:
            # Run inference on batch
            p_images, p_labels = p_data[0].to(device,dtype=torch.float), p_data[1].to(device,dtype=torch.long)
            p_preds, _ = model(p_images)
            # binarize the output of plaque prediction with threshold = 0.5
            # 2d tensor
            p_predicted = (torch.nn.functional.sigmoid(p_preds.data) > 0.5)
            # Record predictions and labels
            predictions["plaque"] = predictions["plaque"] + p_predicted.cpu().data.numpy().tolist()
            labels["plaque"] = labels["plaque"] + p_labels.cpu().data.numpy().tolist()

        # Tissue inference
        for t_data in tissue_loader:
            # Run inference on batch
            t_images, t_labels = t_data[0].to(device,dtype=torch.float), t_data[1].to(device,dtype=torch.long)
            _, t_preds = model(t_images)
            # Get tissue prediction from logit with highest probability
            # 1d tensor
            _, t_predicted = torch.max(t_preds.data, 1)
            # Record predictions and labels
            predictions["tissue"] = predictions["tissue"] + t_predicted.cpu().data.numpy().tolist()
            labels["tissue"] = labels["tissue"] + t_labels.cpu().data.numpy().tolist()

    return predictions, labels

def calc_tissue_metrics(tissue_labels, tissue_predictions, set_name="Validation"):
    """
    Calculate metrics for WMGM data

    Args:
        tissue_labels: list of numpy array actual labels
        tissue_predictions: list of numpy array predicted labels
        set_name: Type of dataset to display in heading
    
    Returns:
        Micro F1 score
        Accuracy score
        Balanced accuracy score
    """
    t_f1_score = f1_score(tissue_labels, tissue_predictions, average='micro', zero_division=0)
    t_acc_score = accuracy_score(tissue_labels, tissue_predictions)
    t_bacc_score = balanced_accuracy_score(tissue_labels, tissue_predictions)

    print("-" * 40)
    print(set_name, "Tissue Dataset:")
    print("Accuracy = ", t_acc_score)
    print("Balanced Accuracy = ", t_bacc_score)
    print("Micro F1 = ", t_f1_score)

    return t_f1_score, t_acc_score, t_bacc_score

def calc_plaque_metrics(plaque_labels, plaque_predictions, set_name="Validation"):
    """
    Calculate metrics for plaque data

    Args:
        plaque_labels: list of numpy array actual labels
        plaque_predictions: list of numpy array predicted labels
        set_name: Type of dataset to display in heading
    
    Returns:
        Sample averaged F1 score
        Sample-wise accuracy score
        Class-averaged balanced accuracy score
    """
    p_f1_score = f1_score(plaque_labels, plaque_predictions, average='samples', zero_division=0)

    p_bacc_scores = np.zeros(3)
    for col in range(3):
        p_bacc_scores[col] = balanced_accuracy_score(plaque_labels[:][col], plaque_predictions[:][col])
    p_bacc_score = np.mean(p_bacc_scores)

    # Calculate sample-wise accuracy to compare with plaque-box paper
    p_predictions = np.array(plaque_predictions)
    p_labels = np.array(plaque_labels)
    p_correct = np.apply_over_axes(np.sum, p_predictions == p_labels, [0,1])
    p_sample_acc_score = float(p_correct) / (p_labels.shape[0] * p_labels.shape[1])

    print("-" * 40)
    print(set_name, "Plaque Dataset:")
    print("Sample-wise Accuracy = ", p_sample_acc_score)
    print("Balanced Accuracy = ", p_bacc_score)
    print("Sample F1 = ", p_f1_score)

    return p_f1_score, p_sample_acc_score, p_bacc_score

def save_metric_fig(metric_data, metric_names, title, filename, base_dir):
    """
    Plot any number of metrics on figure and save to base_dir + filename

    Args:
        metric_data: list of metrics, where each metric is a list of metric values over time
        metric_names: list of metric names to label each plot
        title: title of figure
        filename: name of image file
        base_dir: directory to save images
    """
    fig, _ = plt.subplots()
    # iterate over each metric in metric_data
    for i in range(len(metric_data)):
        # plot (epoch, metric) pair
        plt.plot(metric_data[i], label=metric_names[i])
    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel("Epochs")
    fig.savefig(os.path.join(base_dir, filename))

def save_metric_figs(file_prefix, loss_epoch, acc_train_epoch, acc_val_epoch,
                     bacc_train_epoch, bacc_val_epoch, f1_train_epoch, f1_val_epoch,
                     base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Save loss metrics
    save_metric_fig([loss_epoch["plaque"]], ["loss"], "Plaque Loss", file_prefix+'_plaque_loss.png', base_dir)
    save_metric_fig([loss_epoch["tissue"]], ["loss"], "Tissue Loss", file_prefix+'_tissue_loss.png', base_dir)

    # Save accuracy metrics
    save_metric_fig([acc_train_epoch["plaque"], acc_val_epoch["plaque"]],
                ["train acc", "val acc"], "Plaque Accuracy", file_prefix+'_plaque_acc.png', base_dir)
    save_metric_fig([acc_train_epoch["tissue"], acc_val_epoch["tissue"]],
                ["train acc", "val acc"], "Tissue Accuracy", file_prefix+'_tissue_acc.png', base_dir)

    save_metric_fig([f1_train_epoch["plaque"], f1_val_epoch["plaque"]],
                ["train f1-score", "val f1-score"], "Plaque F1-Score", file_prefix+'_plaque_f1.png', base_dir)
    save_metric_fig([f1_train_epoch["tissue"], f1_val_epoch["tissue"]],
                ["train f1-score", "val f1-score"], "Tissue F1-Score", file_prefix+'_tissue_f1.png', base_dir)

    save_metric_fig([bacc_train_epoch["plaque"], bacc_val_epoch["plaque"]],
                ["train bacc", "val bacc"], "Plaque Balanced Accuracy", file_prefix+'_plaque_bacc.png', base_dir)
    save_metric_fig([bacc_train_epoch["tissue"], bacc_val_epoch["tissue"]],
                ["train bacc", "val bacc"], "Tissue Balanced Accuracy", file_prefix+'_tissue_bacc.png', base_dir)

def find_checkpoint(base_dir, model_name):
    checkpoint_list = glob.glob(os.path.join(base_dir, "checkpoints", f"{model_name}_Epoch_*.pth"))
    max_epoch = 0
    for name in checkpoint_list:
        name_components = name.split("_")
        epoch = name_components[name_components.index("Epoch") + 1]
        max_epoch = max(int(epoch.split(".")[0]), max_epoch)
    return os.path.join(base_dir, "checkpoints", f"{model_name}_Epoch_{max_epoch}.pth")

def fit(model_name, model, optimizer, plaque_train_loader, plaque_val_loader,
        tissue_train_loader, tissue_val_loader, num_epochs, base_dir, load_checkpoint = False,
        stat_count=10, gpu=0, scheduler=None):

    # Move model to device
    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optionally load checkpoint
    if load_checkpoint:
        load_path = find_checkpoint(base_dir, model_name)
        checkpoint = torch.load(load_path, weights_only=False)
        # load model params
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # set metrics based on checkpoint
        curr_model_score = checkpoint['model_score']
        loss_epoch = checkpoint['metrics']['loss_epoch']
        acc_train_epoch = checkpoint['metrics']['acc_train_epoch']
        acc_val_epoch = checkpoint['metrics']['acc_val_epoch']
        f1_val_epoch = checkpoint['metrics']['f1_val_epoch']
        f1_train_epoch = checkpoint['metrics']['f1_train_epoch']
        bacc_val_epoch = checkpoint['metrics']['bacc_val_epoch']
        bacc_train_epoch = checkpoint['metrics']['bacc_train_epoch']

    else:
        start_epoch = 0
        # Init to collect metrics during training
        curr_model_score = 0
        loss_epoch = {"plaque": [], "tissue": []}
        acc_train_epoch = {"plaque": [], "tissue": []}
        acc_val_epoch = {"plaque": [], "tissue": []}
        f1_val_epoch = {"plaque": [], "tissue": []}
        f1_train_epoch = {"plaque": [], "tissue": []}
        bacc_val_epoch = {"plaque": [], "tissue": []}
        bacc_train_epoch = {"plaque": [], "tissue": []}

    # Set loss functions for tissue and plaque data
    plaque_loss_fn = nn.MultiLabelSoftMarginLoss()
    tissue_loss_fn = nn.CrossEntropyLoss()

    # Train until reaching specified number of epochs
    for epoch in range(start_epoch, num_epochs):
        print("Epoch Number", epoch)
        if scheduler:
            print("Current Learning Rate:", scheduler.get_lr()[0])
        start_time = time.time()
        # TRAINING STEP
        batch_losses = train_step(model, optimizer, plaque_train_loader, tissue_train_loader,
                                device, plaque_loss_fn, tissue_loss_fn, epoch, num_epochs, stat_count, scheduler)

        # VALIDATION STEP
        val_metrics = eval_step(model, plaque_val_loader, tissue_val_loader, device)
        val_predictions = val_metrics[0]
        val_labels = val_metrics[1]

        # TRAINING EVALUATION STEP
        train_metrics = eval_step(model, plaque_train_loader, tissue_train_loader, device)
        train_predictions = train_metrics[0]
        train_labels = train_metrics[1]

        print("Epoch time:", time.time() - start_time)
        # Save avg loss
        loss_epoch["plaque"].append(mean(batch_losses["plaque"]))
        loss_epoch["tissue"].append(mean(batch_losses["tissue"]))
        # Step LR scheduler
        scheduler.step()

        # Save tissue validation metrics
        t_val_metrics = calc_tissue_metrics(val_predictions["tissue"], val_labels["tissue"], set_name='Validation')
        f1_val_epoch["tissue"].append(t_val_metrics[0])
        acc_val_epoch["tissue"].append(t_val_metrics[1])
        bacc_val_epoch["tissue"].append(t_val_metrics[2])
        # Save tissue train metrics
        t_train_metrics = calc_tissue_metrics(train_predictions["tissue"], train_labels["tissue"], set_name='Train')
        f1_train_epoch["tissue"].append(t_train_metrics[0])
        acc_train_epoch["tissue"].append(t_train_metrics[1])
        bacc_train_epoch["tissue"].append(t_train_metrics[2])

        # Save plaque validation metrics
        p_val_metrics = calc_plaque_metrics(val_predictions["plaque"], val_labels["plaque"], set_name='Validation')
        f1_val_epoch["plaque"].append(p_val_metrics[0])
        acc_val_epoch["plaque"].append(p_val_metrics[1])
        bacc_val_epoch["plaque"].append(p_val_metrics[2])
        # Save plaque training metrics
        p_train_metrics = calc_plaque_metrics(train_predictions["plaque"], train_labels["plaque"], set_name='Train')
        f1_train_epoch["plaque"].append(p_train_metrics[0])
        acc_train_epoch["plaque"].append(p_train_metrics[1])
        bacc_train_epoch["plaque"].append(p_train_metrics[2])

        # Save checkpoint if avg of balanced accuracy of tissue and plaque improves
        if curr_model_score < bacc_val_epoch["tissue"][-1] * bacc_val_epoch["plaque"][-1]:
            curr_model_score = bacc_val_epoch["tissue"][-1] * bacc_val_epoch["plaque"][-1]
            if not os.path.exists(os.path.join(base_dir, 'checkpoints')):
                os.mkdir(os.path.join(base_dir, 'checkpoints'))
            torch.save(
                {
                    'epoch': epoch,
                    'model_score': curr_model_score,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': {
                        'loss_epoch': loss_epoch,
                        'acc_train_epoch': acc_train_epoch,
                        'acc_val_epoch': acc_val_epoch,
                        'f1_train_epoch': f1_train_epoch,
                        'f1_val_epoch': f1_val_epoch,
                        'bacc_train_epoch': bacc_train_epoch,
                        'bacc_val_epoch': bacc_val_epoch
                    }

                },
                os.path.join(base_dir, 'checkpoints', f"{model_name}_Epoch_{epoch}.pth")
            )
            print("Model checkpoint saved.")
        print("\n")

    # use model name, first epoch idx, and last epoch idx to identify
    plot_file_prefix = f"{model_name}_EpochRange_{start_epoch}_{num_epochs - 1}"
    # After training finishes
    save_metric_figs(plot_file_prefix, loss_epoch, acc_train_epoch, acc_val_epoch,
                     bacc_train_epoch, bacc_val_epoch, f1_train_epoch, f1_val_epoch,
                     os.path.join(base_dir, 'training_plots'))

    return loss_epoch, acc_train_epoch, acc_val_epoch, curr_model_score