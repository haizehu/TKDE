import argparse
import os

def set_args():
    parser = argparse.ArgumentParser('--RADAR')

    # 设置语言参数
    parser.add_argument('--lang', default='python', type=str, help='language')
    # 先解析 lang 参数
    args, _ = parser.parse_known_args()

    # 根据 lang 构建路径
    lang = args.lang
    train_path = os.path.join('./CodeSearchNet-C', lang, 'train.json')
    valid_path = os.path.join('./CodeSearchNet-C', lang, 'valid.json')
    test_path = os.path.join('./CodeSearchNet-C', lang, 'test.json')
    codebase_path = os.path.join('./CodeSearchNet-C', lang, 'codebase.json')

    parser.add_argument('--train_data', default=train_path, type=str, help='train_data path')
    parser.add_argument('--valid_data', default=valid_path, type=str, help='valid_data path')
    parser.add_argument('--test_data', default=test_path, type=str, help='test_data path')
    parser.add_argument("--codebase_file", default=codebase_path, type=str, help="codebase path")

    parser.add_argument('--pretrained_model_path', default='./unixcoder-base', type=str, help='pretrained_model_path') #./roberta_pretrain
    parser.add_argument('--output_dir', default='./outputs_model', type=str, help='output_dir')

    parser.add_argument('--num_train_epochs', default=6, type=int, help='num_train_epochs')

    parser.add_argument('--train_batch_size', default=32, type=int, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='eval_batch_size')

    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning_rate')  # 原2e-5
    parser.add_argument('--seed', default=123456, type=int, help='seed')

    parser.add_argument("--nl_length", default=128, type=int, help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int, help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--context_length", default=20, type=int, help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int, help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument('--hidden_size', type=int, default=768, help="Transformer 隐藏层大小")
    parser.add_argument('--nn_size', type=int, default=251820, help="Hard negative memory bank size")
    # ruby 24927 go 167288 javascript 58025 java 164923 python 251820 php 241241
    parser.add_argument('--nn_k', type=int, default=3, help="Number of hard negatives to mine")
    parser.add_argument('--train_subset_ratio', default=1.0, type=float, help='Ratio of training data to use (e.g., 0.1 for 10%)')

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    return parser.parse_args()