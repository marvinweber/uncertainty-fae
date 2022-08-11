import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

from boneage.rsna_bone_dataloading import RSNABoneageDataset
from boneage.rsna_bone_litmodel import RSNABoneageLitModel
from boneage.rsna_bone_net import resnet18 as resnet18_boneage
from boneage.rsna_bone_net import resnet34 as resnet34_boneage
from util.eval import get_remaining_time
from util.eval_plot_generator import EvalPlotGenerator

RUN_PREDICTIONS = True
SHOW_INTERACTIVE_PLOTS = False
USE_CUDA = True
MC_ITERATIONS = 4
NAME = 'v20_incep_rebal'
# CHECKPOINT_FILE = './lightning_logs/9/checkpoints/epoch=9-val_loss=241.238510.ckpt'  # unscaled
# CHECKPOINT_FILE = './lightning_logs/10/checkpoints/epoch=25-val_loss=0.003713.ckpt'  # rescaled
# CHECKPOINT_FILE = './lightning_logs/11/checkpoints/epoch=5-val_loss=0.004037.ckpt'  # rescaled 500x
# CHECKPOINT_FILE = './lightning_logs/14/checkpoints/epoch=16-val_loss=0.003979.ckpt'  # rescaled 224x loss weighting
CHECKPOINT_FILE = './lightning_logs/20/checkpoints/epoch=25-val_loss=0.003683.ckpt'
LOG_FILE = f'./eval_log_{NAME}_mc_{MC_ITERATIONS}.csv'
EVAL_RESULTS_BASE_DIR = './eval_results'

LOG_COLUMNS = ['image_id', 'boneage', 'predicted_boneage', 'abs_error', 'uncertainty',
               'uncertainty_17_83', 'uncertainty_5_95', 'predictions_median']

# Run Predictions, if requested
if RUN_PREDICTIONS:
    # rsna_boneage_net = resnet18_boneage(pretrained=False, progress=True, num_classes=1)
    rsna_boneage_net = inception_v3(pretrained=False, num_classes=1)
    model = RSNABoneageLitModel.load_from_checkpoint(
        CHECKPOINT_FILE, net=rsna_boneage_net, mc_iterations=MC_ITERATIONS, undo_boneage_rescaling=True)
    model.train(False)

    log_file_h = open(LOG_FILE, 'w', newline='')
    log_file_csv_writer = csv.writer(log_file_h)
    log_file_csv_writer.writerow(LOG_COLUMNS)
    # do not rescale age in dataset (is used for comparison after undoing rescale on prediction)
    data = RSNABoneageDataset('./images/with_preprocessed_val_annotations.csv',
                              rescale_boneage=False, target_dimensions=(299,299))
    test_dataloader = DataLoader(data, batch_size=1, shuffle=False)
    if USE_CUDA:
        model.cuda()

    start = datetime.now()

    for batch, (X, y) in enumerate(test_dataloader):
        if USE_CUDA:
            X = X.cuda()

        prediction, metrics = model.forward_with_mc(X)

        boneage = y.detach().numpy()[0]
        prediction = prediction.cpu().detach().numpy()

        abs_error = abs(boneage - prediction)
        uncertainty = metrics['uncertainty'].detach().numpy()
        log_file_csv_writer.writerow([
            'unkown',
            boneage,
            prediction,
            abs_error,
            uncertainty,
            metrics['uncertainty_17_83_range'].detach().numpy(),
            metrics['uncertainty_5_95_range'].detach().numpy(),
            metrics['median'].detach().numpy(),
        ])

        if (batch + 1) % 50 == 0:
            percentage_done, minutes_done, minutes_remaining, expected_end, expected_minutes = \
                get_remaining_time(batch+1, len(test_dataloader), start)
            print(f'Progress {batch+1}/{len(test_dataloader)} -- {percentage_done} % -- '
                  f'{minutes_done}/{expected_minutes} Minutes -- '
                  f'Expected End: {expected_end.strftime("%y-%m-%d %H:%M")} '
                  f'In {minutes_remaining} Minutes.')

    log_file_h.close()


# EVALUATION
log_df = pandas.read_csv(LOG_FILE)
eval_results_dir = os.path.join(EVAL_RESULTS_BASE_DIR, NAME)
plot_generator = EvalPlotGenerator(log_df, eval_results_dir, img_prepend_str=NAME)
os.makedirs(eval_results_dir, exist_ok=True)

print('########### STATS ##########')
print(f'MIN PREDICTION:   {log_df["predicted_boneage"].min()}')
print(f'MAX PREDICTION:   {log_df["predicted_boneage"].max()}')
print(f'MIN ABS ERROR:    {log_df["abs_error"].min()}')
print(f'MAX ABS ERROR:    {log_df["abs_error"].max()}')
print(f'MEAN ABS ERROR:   {log_df["abs_error"].mean()}')
print(f'MEDIAN ABS ERROR: {log_df["abs_error"].median()}')
print(f'MIN UNCERTAINTY:  {log_df["uncertainty"].min()}')
print(f'MAX UNCERTAINTY:  {log_df["uncertainty"].max()}')
print('########### STATS ##########')

plot_generator.plot_abs_error_uncertainty_scatter()
plot_generator.plot_abs_error_uncertainty_17_83_scatter()
plot_generator.plot_bonage_distribution()
plot_generator.plot_uncertainty_by_boneage()
plot_generator.plot_uncertainty_comparison()
plot_generator.plot_abs_error_by_boneage()
plot_generator.plot_uncertainty_by_abs_error()
plot_generator.plot_prediction_vs_truth()
plot_generator.plot_tolerated_uncertainty_abs_error()

if SHOW_INTERACTIVE_PLOTS:
    plt.show()
