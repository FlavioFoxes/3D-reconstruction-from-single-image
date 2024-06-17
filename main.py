import argparse

from src.train.trainer import trainer
from src.test.tester import tester
from src.examples.plot_gt import plot_ground_truth
from src.examples.plot_pred import plot_model_prediction
from src.examples.example import plot_example


def main():
    parser = argparse.ArgumentParser(description='Train or test the neural network')

    parser.add_argument('--train',
                        help='trains the network',
                        action='store_true')
    parser.add_argument('--pretrain',
                        help='trains a pretrained network',
                        action='store_true')
    
    parser.add_argument('--test',
                        help='trains the network',
                        action='store_true')
    
    parser.add_argument('--ground', 
                        help='Plot an example of ground truth',
                        type=int)
    parser.add_argument('--pred', 
                        help='Plot an example of prediction',
                        type=int)
    parser.add_argument('--example', 
                        help='Plot an example of comparison between ground truth and model prediction',
                        type=int)

    args = parser.parse_args()

    if args.train and args.test:
        parser.error("Specify just one argument.")

    if args.train:
        trainer()
    elif args.pretrain:
        trainer(isPretrained=True)
    elif args.test:
        tester()
    elif args.ground is not None:
        plot_ground_truth(args.ground)
    elif args.pred is not None:
        plot_model_prediction(args.pred)
    elif args.example is not None:
        plot_example(args.example)
    else:
        parser.error("Specify at least one argument.")


if __name__ == "__main__":
    main()