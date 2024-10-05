import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train or Test ViT Autoencoder")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help="Specify whether to train or test the model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to the checkpoint to load for training or testing")
    parser.add_argument('--train_data_path', type=str, default='../../dataSet/classified_kraftig/train',
                        help="Path to training data")
    parser.add_argument('--test_data_path', type=str, default='../../dataSet/classified_kraftig/test',
                        help="Path to testing data")
    parser.add_argument('--model_save_path', type=str, default='../saved_model/auto_ViT',
                        help="Path to save model checkpoints")
    return parser.parse_args()
