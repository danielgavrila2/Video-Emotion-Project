import os


# AWS Sagemaker

# TODO: Implement AWS Sagemaker infrastructure

SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING')
