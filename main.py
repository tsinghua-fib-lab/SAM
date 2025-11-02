import torch
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')  # suppress warnings
from trainer import create_trainer


def main():
	# argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=400, help='num training epochs')
	parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
	parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
	parser.add_argument('--dataset', type=str, default='Beauty', help='dataset name')
	parser.add_argument('--item_dim', type=int, default=256, help='ID embedding dimension')
	parser.add_argument('--dropout', type=float, default=0.2, help='dropout for ID positional encoding')
	parser.add_argument('--stage2_dropout', type=float, default=0.2, help='dropout inside Stage2 attention/FFN')
	parser.add_argument('--max_seq_len', type=int, default=20, help='max sequence length')
	parser.add_argument('--min_seq_len', type=int, default=3, help='min sequence length')
	parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
	parser.add_argument('--lr_step_size', type=int, default=10, help='StepLR step size')
	parser.add_argument('--lr_gamma', type=float, default=0.8, help='StepLR gamma')
	parser.add_argument('--early_stopping_patience', type=int, default=10, help='early stop patience (epochs without improvement)')
	parser.add_argument('--num_interest_fusion_layers', type=int, default=1, help='Transformer layers (Stage2)')
	parser.add_argument('--stage2_num_heads', type=int, default=4, help='multi-head attention heads (Stage2)')
	
	# parse arguments
	args = parser.parse_args()  
	
	# device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# config dict
	config = {
		'batch_size': args.batch_size,
		'learning_rate': args.learning_rate,
		'weight_decay': args.weight_decay,
		'data_dir': './',
		'max_seq_len': args.max_seq_len,
		'min_seq_len': args.min_seq_len,
		'num_workers': args.num_workers,
		'item_dim': int(args.item_dim),
		'interest_dim': 768,
		'dropout': float(args.dropout),
		'stage2_dropout': float(args.stage2_dropout),
		'lr_step_size': args.lr_step_size,
		'lr_gamma': args.lr_gamma,
		'interests_pt_path': f"./Interests/interests_{args.dataset}.pt",
		'txt_path': f"./Data/{args.dataset}.txt",
		'early_stopping_patience': int(args.early_stopping_patience),
		'num_interest_fusion_layers': args.num_interest_fusion_layers,
		'stage2_num_heads': args.stage2_num_heads
	}

	# create model & trainer
	trainer = create_trainer(config, device)
		
	# train
	history = trainer.train(num_epochs=args.epochs)
		
	if history:
		final_loss = history['train_loss'][-1] if 'train_loss' in history else 0
		print(f"Final training loss: {final_loss:.4f}")
	
if __name__ == "__main__":
	exit_code = main()
	sys.exit(exit_code)