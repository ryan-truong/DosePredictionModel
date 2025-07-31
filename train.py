
import os
import argparse
from model import Model
import lightning as L
from lightning.pytorch.loggers.neptune import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
from neptune import ANONYMOUS_API_TOKEN # Placeholder for Neptune API token
from myLightningUtils import CustomWriter, BrachyDataModule
from model import Model  # Import the custom model architecture
import lightning as L
from myLightningUtils import CustomWriter, BrachyDataModule  # Custom data module for brachytherapy data

def main(args):
    # Get current directory for file paths
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Setup Neptune logger for experiment tracking
    neptune_logger = NeptuneLogger(
        # api_key=ANONYMOUS_API_TOKEN,
        api_token="",  # API token needs to be filled
        project="",    # Project name needs to be filled
        tags=[""],     # Tags for experiment organization
        source_files=[current_dir+"/*.py", "config.yaml"],  # Track source code and config
        description="Description here!",  # Experiment description
        dependencies="infer",  # Automatically detect dependencies
        log_model_checkpoints=False  # Don't upload model checkpoints to Neptune
    )

    # Create unique output directory
    project_name = ''
    duplicate_count = 0
    while os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Output_' + project_name + "_" + str(duplicate_count))):
        duplicate_count += 1
    outpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Output_'+ project_name + "_" + str(duplicate_count))
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    # Handle Slurm directory path for cluster environments
    slurm_dir = args.slurm_dir
    if os.name == "nt":  # If running on Windows
        slurm_dir = None

    # Setup checkpoint callback to save best models
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(outpath,'checkpoints'),
        filename='BrachyNet-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,  # Save the top 3 models with lowest validation loss
        mode='min',
    )

    # Setup early stopping to prevent overfitting
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,  # Stop training if no improvement for 100 epochs
        verbose=True,
        mode='min'
    )

    # Initialize the model with specified architecture parameters
    CascadeUnet = Model(in_ch=11, out_ch=1,
                                    list_ch_A=[-1, 24, 48, 94, 192, 768],
                                    list_ch_B=[-1, 24, 48, 94, 192, 768],
                                    dropout_prob=0.1,
                                    train_batch_size=args.batch_size,
                                    val_batch_size=args.batch_size,
                                    ).float()

    # Setup data module for handling training and validation data
    brachy_data_module = BrachyDataModule(
        data_dir = args.slurm_dir,
        batch_size = args.batch_size,
        num_workers = 20,  # Number of workers for data loading
    )

    # Configure the PyTorch Lightning trainer
    trainer = L.Trainer(
        accelerator="gpu",  # Use GPU for training
        logger=neptune_logger,  # Use Neptune for logging
        default_root_dir=outpath,
        log_every_n_steps=1,  # Log metrics every step
        max_epochs=-1,  # Train indefinitely (will be stopped by early stopping)
        callbacks=[checkpoint_callback, early_stop_callback],  # Register callbacks
        sync_batchnorm=True,  # Synchronize batch normalization for multi-GPU
    )

    # Log model architecture to Neptune
    neptune_logger.log_model_summary(CascadeUnet)

    # Begin model training
    trainer.fit(model=CascadeUnet, datamodule=brachy_data_module)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training (default: 1)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[3,2,1,0],
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--slurm_dir', type=str,default='/scratch',
                        help='Directory path for Slurm environment')

    args = parser.parse_args()
    main(args)
