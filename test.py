import torch
import numpy as np

from model import Discriminator
from utils import set_model_mode


def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode):
    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    if training_mode == 'DANN':
        discriminator.cuda()
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Process source and target data
        source_image, source_label = process_data(source_data, expand_channels=True)
        target_image, target_label = process_data(target_data)

        # Compute source and target predictions
        source_pred = compute_output(encoder, classifier, source_image, alpha=None)
        target_pred = compute_output(encoder, classifier, target_image, alpha=None)

        # Update correct counts
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).sum().item()
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).sum().item()

        if training_mode == 'DANN':
            # Process combined images for domain classification
            combined_image = torch.cat((source_image, target_image), 0)
            domain_labels = torch.cat((torch.zeros(source_label.size(0), dtype=torch.long),
                                       torch.ones(target_label.size(0), dtype=torch.long)), 0).cuda()

            # Compute domain predictions
            domain_pred = compute_output(encoder, discriminator, combined_image, alpha=alpha)
            domain_correct += domain_pred.eq(domain_labels.data.view_as(domain_pred)).sum().item()

    source_dataset_len = len(source_test_loader.dataset)
    target_dataset_len = len(target_test_loader.dataset)

    accuracies = {
        "Source": {
            "correct": source_correct,
            "total": source_dataset_len,
            "accuracy": calculate_accuracy(source_correct, source_dataset_len)
        },
        "Target": {
            "correct": target_correct,
            "total": target_dataset_len,
            "accuracy": calculate_accuracy(target_correct, target_dataset_len)
        }
    }

    if training_mode == 'DANN':
        accuracies["Domain"] = {
            "correct": domain_correct,
            "total": source_dataset_len + target_dataset_len,
            "accuracy": calculate_accuracy(domain_correct, source_dataset_len + target_dataset_len)
        }

    print_accuracy(training_mode, accuracies)


def process_data(data, expand_channels=False):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    if expand_channels:
        images = images.repeat(1, 3, 1, 1)  # Repeat channels to convert to 3-channel images
    return images, labels


def compute_output(encoder, classifier, images, alpha=None):
    features = encoder(images)
    if isinstance(classifier, Discriminator):
        outputs = classifier(features, alpha)  # Domain classifier
    else:
        outputs = classifier(features)  # Category classifier
    preds = outputs.data.max(1, keepdim=True)[1]
    return preds


def calculate_accuracy(correct, total):
    return 100. * correct / total


def print_accuracy(training_mode, accuracies):
    print(f"Test Results on {training_mode}:")
    for key, value in accuracies.items():
        print(f"{key} Accuracy: {value['correct']}/{value['total']} ({value['accuracy']:.2f}%)")
