import sys
import time

from config import set_args
from utils import convert_examples_to_features,convert_examples_to_features_base

import os
import torch
import random
import json
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm
from model import Model
from membank import VecIndexBank
from datetime import datetime
from torch.optim import AdamW
from torch import nn
from torch import amp
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler

import subprocess

# 设置空闲显存阈值（单位：MB）
MEMORY_FREE_THRESHOLD_MB = 40960  # 例如至少空出 13GB 才启动任务

def get_gpu0_free_memory():
    """返回 GPU 0 的空闲显存（MB）"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(output.strip().split('\n')[0])
    except Exception as e:
        print("无法获取 GPU 显存:", e)
        return 0

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        # 取语言名（倒数第二个目录）
        lang = file_path.split('/')[-2]
        # 取文件名前缀
        prefix = file_path.split('/')[-1][:-5]
        # 拼接缓存文件名
        cache_file = os.path.join(args.output_dir, f"{lang}_{prefix}-C.pkl")
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data), desc="Processing examples"))
            pickle.dump(self.examples, open(cache_file, 'wb'))
        # --- 添加一个列表，用于快速索引 ---
        self.all_examples_by_index = list(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (item,
                torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].package_ids),
                torch.tensor(self.examples[item].function_ids),
                torch.tensor(self.examples[item].levels))

    # --- 添加一个辅助方法来批量检索硬负样本 ---
    def get_batch_by_idx(self, indexs):
        """
        根据原始数据索引列表，检索并打包成一个新的批次。
        indexs: [bs * k]
        """
        batch_items = [self.all_examples_by_index[i] for i in indexs]

        # 重新打包成 tensors
        code_ids = torch.stack([torch.tensor(ex.code_ids) for ex in batch_items])
        nl_ids = torch.stack([torch.tensor(ex.nl_ids) for ex in batch_items])
        package_ids = torch.stack([torch.tensor(ex.package_ids) for ex in batch_items])
        function_ids = torch.stack([torch.tensor(ex.function_ids) for ex in batch_items])
        levels = torch.stack([torch.tensor(ex.levels) for ex in batch_items])
        # 注意：这里的 'indexs' 是假的，我们只需要数据
        return (indexs, code_ids, nl_ids, package_ids, function_ids, levels)


class TextDatasetBase(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        # 取语言名（倒数第二个目录）
        lang = file_path.split('/')[-2]
        # 取文件名前缀
        prefix = file_path.split('/')[-1][:-5]
        # 拼接缓存文件名
        cache_file = os.path.join(args.output_dir, f"{lang}_{prefix}.pkl")
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features_base, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].nl_ids))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def write_arg_log(s, output_dir):
    logs_path = os.path.join(output_dir, 'arg_log.txt')
    with open(logs_path, 'a+', encoding='utf-8') as f:
        f.write(f"\n==== Log at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        for key in sorted(s.keys()):
            f.write(f"{key} : {s[key]}\n")
    print(f"日志文件 {logs_path} 已创建成功。")

class_loss_fn = nn.CrossEntropyLoss()

def hfedr_loss_one(nl_vecs, code_vecs, temp=0.05):
    B, H = nl_vecs.shape
    # 1. 计算相似度矩阵 [B, B]
    # 行代表 query (NL)，列代表 key (Code)
    scores = (nl_vecs @ code_vecs.T) / temp
    # 2. 生成对角线标签 [0, 1, ..., B-1]
    labels = torch.arange(B, device=nl_vecs.device)
    # 3. 计算 NL -> Code 的 Loss (即原有的 Loss)
    # F.cross_entropy 默认对 dim=1 (行) 做 Softmax
    loss_nl_to_code = F.cross_entropy(scores, labels)
    # 4. 计算 Code -> NL 的 Loss (新增的反向 Loss)
    # 将 scores 转置，使得行代表 Code，列代表 NL
    # loss_code_to_nl = F.cross_entropy(scores.T, labels)
    # # 5. 双向 Loss 取平均
    # loss = 0.5 * (loss_nl_to_code + loss_code_to_nl)
    return loss_nl_to_code

def hfedr_loss_two(nl_vecs, code_vecs, temp=0.05):
    B, H = nl_vecs.shape
    # 1. 计算相似度矩阵 [B, B]
    # 行代表 query (NL)，列代表 key (Code)
    scores = (nl_vecs @ code_vecs.T) / temp
    # 2. 生成对角线标签 [0, 1, ..., B-1]
    labels = torch.arange(B, device=nl_vecs.device)
    # 3. 计算 NL -> Code 的 Loss (即原有的 Loss)
    # F.cross_entropy 默认对 dim=1 (行) 做 Softmax
    loss_nl_to_code = F.cross_entropy(scores, labels)
    # 4. 计算 Code -> NL 的 Loss (新增的反向 Loss)
    # 将 scores 转置，使得行代表 Code，列代表 NL
    loss_code_to_nl = F.cross_entropy(scores.T, labels)
    # 5. 双向 Loss 取平均
    loss = 0.5 * (loss_nl_to_code + loss_code_to_nl)
    return loss

def single_loss(nl_vec, code_vec, nl_neighbor, temp=0.05):
    # nl_vec: [bs, dim] (正样本)
    # code_vec: [bs, dim] (锚点)
    # nl_neighbor: [bs, k, dim] (硬负样本)

    bs, dim = nl_vec.shape
    # 将正样本 [bs, 1, dim] 和硬负样本 [bs, k, dim] 连接起来
    # nl_new_vec: [bs, 1+k, dim]
    nl_new_vec = torch.cat([nl_vec.unsqueeze(1), nl_neighbor], dim=1)

    # (锚点) * (正样本 + 硬负样本)
    # sim_matrix: [bs, 1+k]
    sim_matrix = torch.einsum("bhd,bd->bh", nl_new_vec, code_vec) # 使用 einsum 批量计算点积
    sim_matrix = sim_matrix / temp
    # 正样本总是在索引 0
    labels = torch.zeros(bs, device=nl_vec.device, dtype=torch.long)
    return class_loss_fn(sim_matrix, labels)

def neighbor_loss_1(nl_vec, code_vec, code_neighbor):
    # 损失1：锚点=NL, 正样本=Code, 硬负=Code_Neighbors
    loss_1 = single_loss(code_vec, nl_vec, code_neighbor)
    # 损失2：锚点=Code, 正样本=NL, 硬负=NL_Neighbors
    # loss_2 = single_loss(nl_vec, code_vec, nl_neighbor)

    return loss_1

def intra_hard_loss_bidirectional(orig_vec, aug_vec, hard_negs, temp=0.05):

    # score(Aug, Orig) vs score(Negs, Orig)
    loss_fwd = single_loss(aug_vec, orig_vec, hard_negs, temp)
    # score(Orig, Aug) vs score(Negs, Aug)
    loss_bwd = single_loss(orig_vec, aug_vec, hard_negs, temp)

    return 0.5 * (loss_fwd + loss_bwd)

def get_multilang_keywords_tensor(tokenizer, device):
    LANG_KEYWORDS = {
        'python': {
            'def', 'return', 'class', 'if', 'else', 'elif', 'for', 'while', 'import', 'from',
            'try', 'except', 'raise', 'finally', 'with', 'as', 'assert',
            'and', 'or', 'not', 'is', 'in', 'None', 'True', 'False', 'lambda', 'global',
            'self', 'yield', 'del', 'async', 'await'  # 补充
        },
        'java': {
            'public', 'private', 'protected', 'class', 'interface', 'abstract', 'void', 'static',
            'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
            'import', 'package', 'new', 'try', 'catch', 'finally', 'throw', 'throws',
            'true', 'false', 'null', 'this', 'super', 'instanceof', 'extends', 'implements',
            'int', 'boolean', 'char', 'double', 'float', 'long', 'byte', 'short'
        },
        'go': {
            'func', 'var', 'const', 'type', 'struct', 'interface', 'package', 'import', 'return',
            'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'break', 'continue',
            'go', 'defer', 'chan', 'map', 'select', 'goto', 'true', 'false', 'nil',
            'int', 'int64', 'float64', 'string', 'bool', 'byte', 'error'
        },
        'javascript': {
            'function', 'var', 'const', 'let', 'class', 'return', 'if', 'else', 'switch', 'case',
            'for', 'while', 'do', 'break', 'continue', 'import', 'export', 'default', 'from',
            'try', 'catch', 'finally', 'throw', 'async', 'await', 'new', 'this',
            'true', 'false', 'null', 'undefined', 'typeof', 'instanceof', 'void', 'delete',
            'debugger', 'extends', 'super'
        },
        'php': {
            'function', 'class', 'interface', 'trait', 'extends', 'implements',
            'public', 'private', 'protected', 'static', 'final', 'abstract',
            'return', 'if', 'else', 'elseif', 'foreach', 'for', 'while', 'switch', 'case',
            'echo', 'use', 'namespace', 'require', 'include', 'new', 'clone',
            'try', 'catch', 'finally', 'throw', 'true', 'false', 'null', 'array',
            'global', 'var', 'const', 'exit', 'die'
        },
        'ruby': {
            'def', 'end', 'class', 'module', 'return', 'if', 'else', 'elsif', 'unless',
            'while', 'until', 'for', 'break', 'next', 'redo', 'retry', 'do', 'yield',
            'require', 'include', 'extend', 'begin', 'rescue', 'ensure', 'raise',
            'true', 'false', 'nil', 'self', 'super', 'alias', 'defined?',
            'then', 'when', 'case'
        }
    }

    # ================= 2. 结构定界符 (Topology) =================
    # 定义代码的层级和范围
    STRUCTURAL_PUNCTUATION = {
        '{', '}', '(', ')', '[', ']',  # 括号
        ';', ',', '.', ':',  # 分割符
        '@',  # Java/Python注解
        '$',  # PHP变量前缀
        '?',  # 三元运算/泛型
    }

    # ================= 3. 逻辑运算符 (Logic Flow) =================
    # 定义代码的运算逻辑，防止 Mask 导致逻辑歧义
    LOGICAL_OPERATORS = {
        '=', '+', '-', '*', '/', '%',  # 算术
        '==', '!=', '<', '>', '<=', '>=',  # 比较
        '&&', '||', '!', '&', '|', '^', '~',  # 逻辑/位运算
        '+=', '-=', '*=', '/=',  # 复合赋值
        '->', '=>', '::', '...',  # 指针/箭头/域操作
        ':=', '<<', '>>'  # Go/位移
    }

    # 合并所有保护对象
    all_protected_tokens = set()
    for lang in LANG_KEYWORDS:
        all_protected_tokens.update(LANG_KEYWORDS[lang])

    all_protected_tokens.update(STRUCTURAL_PUNCTUATION)
    all_protected_tokens.update(LOGICAL_OPERATORS)

    # 转为 Token IDs (处理 BPE 前缀空格问题)
    protected_ids = set()
    for token in all_protected_tokens:
        # 1. 原始 token
        ids = tokenizer.encode(token, add_special_tokens=False)
        protected_ids.update(ids)

        # 2. 带空格前缀的 token (Roberta Tokenizer 特性)
        # 很多符号前面可能有空格，比如 " = "
        ids_prefix = tokenizer.encode(' ' + token, add_special_tokens=False)
        protected_ids.update(ids_prefix)

    # 转为 Tensor
    protected_tensor = torch.tensor(list(protected_ids), dtype=torch.long, device=device)

    print(f"Total protected tokens count (vocab level): {len(protected_ids)}")
    return protected_tensor

def augment_data(input_ids, tokenizer, protected_ids=None, span_len=3, mlm_probability=0.2):
    input_ids = input_ids.clone()
    device = input_ids.device
    B, L = input_ids.shape

    # 1. 初始化概率矩阵
    # 动态补偿
    seed_prob = mlm_probability / span_len
    probability_matrix = torch.full(input_ids.shape, seed_prob, device=device)


    probability_matrix[:, :3] = 0.0

    special_ids_list = tokenizer.all_special_ids
    special_ids_tensor = torch.tensor(special_ids_list, device=device)

    special_mask = torch.isin(input_ids, special_ids_tensor)
    probability_matrix.masked_fill_(special_mask, 0.0)

    keyword_mask = None
    if protected_ids is not None:
        keyword_mask = torch.isin(input_ids, protected_ids)
        probability_matrix.masked_fill_(keyword_mask, 0.0)

    # 2. 采样种子点
    seed_indices = torch.bernoulli(probability_matrix).bool()

    # 连续片段掩码 (Span Masking) ===

    # 通过滚动操作实现 Span 扩散
    masked_indices = seed_indices.clone()
    for i in range(1, span_len):
        masked_indices |= torch.roll(seed_indices, shifts=i, dims=1)

    masked_indices[:, :3] = False  # 切除头部
    masked_indices &= ~special_mask  # 切除特殊字符
    if keyword_mask is not None:
        masked_indices &= ~keyword_mask  # 切除关键字

    # === Layer 3: 执行增强 (混合策略 Mixed Strategy) ===

    # 80% Identity Rule
    indices_to_change = torch.bernoulli(torch.full((B, L), 0.8, device=device)).bool() & masked_indices

    # 生成一个策略概率矩阵，决定具体怎么改
    strategy_probs = torch.rand((B, L), device=device)
    # 80% 的情况变 MASK
    mask_mask = indices_to_change & (strategy_probs < 0.8)
    # 20% 的情况变 Random Token
    replace_mask = indices_to_change & (strategy_probs >= 0.8)
    # 应用 MASK
    if mask_mask.any():
        input_ids[mask_mask] = tokenizer.mask_token_id
    # 应用 Random Replacement
    if replace_mask.any():
        vocab_size = tokenizer.vocab_size
        random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=device)
        # 防止随机出的噪声恰好是特殊 Token
        random_is_special = torch.isin(random_tokens, special_ids_tensor)
        random_tokens[random_is_special] = tokenizer.mask_token_id  # 兜底换成 Mask
        input_ids[replace_mask] = random_tokens[replace_mask]

    return input_ids


def load_checkpoint(model, optimizer, scheduler, load_path, device=None):
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    print(f"Loaded checkpoint from {load_path}, resume from epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch


def train_model(args, model, tokenizer, pool):
    # 构建数据集和加载器
    train_dataset = TextDataset(tokenizer, args, args.train_data, pool=pool)

    # ================= 低资源场景切分 =================
    if args.train_subset_ratio < 1.0:
        total_len = len(train_dataset.examples)
        subset_len = int(total_len * args.train_subset_ratio)

        print(f"\n[Low-Resource Mode] Original Training Data: {total_len}")
        print(f"[Low-Resource Mode] Slicing to top {args.train_subset_ratio * 100}%: {subset_len} examples")

        # 为了保证实验可复现，使用固定随机种子打乱后切分
        rng = random.Random(args.seed)
        rng.shuffle(train_dataset.examples)

        # 执行切分
        train_dataset.examples = train_dataset.examples[:subset_len]

        if hasattr(train_dataset, 'all_examples_by_index'):
            train_dataset.all_examples_by_index = list(train_dataset.examples)

        # 必须更新 Memory Bank 的大小
        args.nn_size = subset_len
        print(f"[Low-Resource Mode] Memory Bank Size (nn_size) auto-adjusted to: {args.nn_size}\n")
    # ==============================================================

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    # 计算训练总步数
    total_steps = len(train_dataloader) * args.num_train_epochs
    # 设置 warmup 步数
    warmup_steps = int(0.05 * total_steps)

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 多卡训练支持
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(args.device)

    # --- 初始化内存库 ---
    print("Initializing Memory Banks for Hard Negative Mining...")
    nl_bank = VecIndexBank(size=args.nn_size, K=args.nn_k, dim=args.hidden_size).to(args.device)
    code_bank = VecIndexBank(size=args.nn_size, K=args.nn_k, dim=args.hidden_size).to(args.device)
    # --- 结束 ---

    # 初始化保存路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_model_dir = os.path.join(args.output_dir, "best_mrr_model")
    os.makedirs(best_model_dir, exist_ok=True)

    bestmodel = "model_epoch_.pt"
    checkpoint_path = os.path.join(args.output_dir, bestmodel)

    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device=args.device
        )
        print(f"从第 {start_epoch} 轮开始继续训练")
    else:
        print("没有找到 checkpoint，从头开始训练")
        start_epoch = 0

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Total train batch size = {args.train_batch_size}")
    print(f"  Total optimization steps = {len(train_dataloader) * args.num_train_epochs}")

    # AMP混合精度初始化
    scaler = amp.GradScaler("cuda")
    optimizer.zero_grad()
    model.train()

    tr_loss, tr_loss_bcl, tr_loss_hncl, tr_loss_intra, tr_loss_intra_hard = 0.0, 0.0, 0.0, 0.0, 0.0
    tr_num = 0

    print("正在构建多语言关键字保护集合...")
    keywords_tensor = get_multilang_keywords_tensor(tokenizer, args.device)
    for epoch in range(start_epoch, args.num_train_epochs):

        # ==================================
        # 统一 K 值和预热策略
        shared_k = min(args.nn_k, max(1, epoch + 1))
        tqdm.write(f"Epoch {epoch + 1}: Shared K={shared_k}")
        # =============================================

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            indexs, code_ids, nl_ids, package_ids, function_ids, levels = batch
            indexs = indexs.to(args.device)
            code_ids = code_ids.to(args.device)
            nl_ids = nl_ids.to(args.device)
            package_inputs = package_ids.to(args.device)
            function_inputs = function_ids.to(args.device)
            # levels 等无需传 device，model 内部处理即可

            B, H = nl_ids.shape[0], args.hidden_size

            with amp.autocast("cuda"):
                code_vecs = model(code_inputs=code_ids)
                nl_vecs = model(nl_inputs=nl_ids, context_inputs=[package_inputs, function_inputs], levels=levels)
                loss_bcl = hfedr_loss_one(nl_vecs, code_vecs)

                # --- 更新内存库 ---
                nl_bank.put(nl_vecs.detach().float(), indexs)
                code_bank.put(code_vecs.detach().float(), indexs)

                # 保护6种语言的关键字
                aug_code_ids = augment_data(
                    code_ids,
                    tokenizer,
                    protected_ids=keywords_tensor,  # 传入关键字 Tensor
                    span_len=4,  # Span=4 覆盖长变量名
                    mlm_probability=0.3
                )

                # 策略：不保护普通单词，逼模型去猜被盖住的短语 (Span=2)
                aug_nl_ids = augment_data(
                    nl_ids,
                    tokenizer,
                    protected_ids=None,  # Query 不需要保护关键字
                    span_len=2,  # Span=2 覆盖短语
                    mlm_probability=0.15  # 0.15
                )

                # 2.1 计算增强数据的向量
                aug_code_vecs = model(code_inputs=aug_code_ids)
                aug_nl_vecs = model(nl_inputs = aug_nl_ids, context_inputs = [package_inputs, function_inputs], levels = levels)

                if shared_k > 0:
                    # 1. 设置 Bank 的 K 值
                    nl_bank.K = shared_k
                    nl_bank.vec_bank.K = shared_k
                    code_bank.K = shared_k
                    code_bank.vec_bank.K = shared_k

                    # =====================================================
                    # Part 跨模态硬负样本
                    # =====================================================

                    # NL 搜 Code Bank
                    code_neighbor_idxs, code_neighbor_bank_idxs = code_bank.get_index(
                        nl_vecs.detach().float(), indexs
                    )
                    # Code 搜 NL Bank (为 HNCL 准备)xs, nl_neighbor_bank_idxs = nl_bank.get_index(
                    #     code_vecs.detach().float(), indexs
                    # )

                    # 取数据
                    hncl_code_batch = train_dataset.get_batch_by_idx(code_neighbor_idxs)
                    # hncl_nl_batch = train_dataset.get_batch_by_idx(nl_neighbor_idxs)

                    # 前向传播 (计算梯度)
                    # 注意：我们要取 Code batch 里的 Code, NL batch 里的 NL
                    (_, hn_code_ids, _, _, _, _) = hncl_code_batch
                    # (_, _, hn_nl_ids) = hncl_nl_batch

                    hn_code_vecs = model(code_inputs=hn_code_ids.to(args.device))
                    # hn_nl_vecs = model(nl_inputs=hn_nl_ids.to(args.device))

                    # 更新 Bank (用最新的干净向量)
                    code_bank.update(code_neighbor_bank_idxs, hn_code_vecs.detach().float())
                    # nl_bank.update(nl_neighbor_bank_idxs, hn_nl_vecs.detach().float())

                    # 计算 HNCL Loss
                    clean_code_neg = hn_code_vecs.view(B, shared_k, H)
                    # clean_nl_neg = hn_nl_vecs.view(B, shared_k, H)
                    loss_hncl = neighbor_loss_1(nl_vecs, code_vecs, clean_code_neg)

                    # =====================================================
                    # 自模态硬负样本
                    # =====================================================

                    # --- 1. Code 侧自查 ---
                    # 用 Code 搜 Code Bank (找结构相似的代码)
                    intra_code_idxs, _ = code_bank.get_index(
                        code_vecs.detach().float(), indexs
                    )
                    # 取出这些难样本的 CODE 部分
                    intra_code_batch = train_dataset.get_batch_by_idx(intra_code_idxs)
                    (_, intra_hard_code_ids, _, _, _, _) = intra_code_batch

                    # 计算向量
                    intra_hard_code_vecs = model(code_inputs=intra_hard_code_ids.to(args.device))
                    intra_hard_code_neg = intra_hard_code_vecs.view(B, shared_k, H)

                    # 计算 Code 侧硬内功 Loss
                    # Anchor: Code, Positive: AugCode, Negative: Hard_Code_Neighbors
                    loss_intra_code_hard = intra_hard_loss_bidirectional(code_vecs, aug_code_vecs, intra_hard_code_neg)

                    # --- 2. Query (NL) 侧自查 ---
                    # 用 NL 搜 NL Bank (找语义相似的查询)
                    intra_nl_idxs, _ = nl_bank.get_index(
                        nl_vecs.detach().float(), indexs
                    )
                    # 取出这些难样本的 NL 部分
                    intra_nl_batch = train_dataset.get_batch_by_idx(intra_nl_idxs)
                    (_, _, intra_hard_nl_ids, intra_hard_package_inputs, intra_hard_function_inputs, intra_hard_levels) = intra_nl_batch
                    # 计算向量
                    intra_hard_nl_vecs = model(
                        nl_inputs=intra_hard_nl_ids.to(args.device),
                        context_inputs=[
                            intra_hard_package_inputs.to(args.device),
                            intra_hard_function_inputs.to(args.device)  # 修改这里
                        ],
                        levels=intra_hard_levels  # 修改这里
                    )
                    intra_hard_nl_neg = intra_hard_nl_vecs.view(B, shared_k, H)

                    # 计算 NL 侧硬内功 Loss
                    # Anchor: NL, Positive: AugNL, Negative: Hard_NL_Neighbors
                    loss_intra_nl_hard = intra_hard_loss_bidirectional(nl_vecs, aug_nl_vecs, intra_hard_nl_neg)

                    # --- 3. 组合 IntraHard Loss ---
                    # 建议权重稍微给高一点点，因为这部分梯度质量很高
                    loss_intra_hard = 0.1 * (loss_intra_code_hard + loss_intra_nl_hard)

                    # 计算软内功 (Batch内) 作为补充
                    loss_intra_nl_batch = hfedr_loss_two(nl_vecs, aug_nl_vecs)
                    loss_intra_code_batch = hfedr_loss_two(code_vecs, aug_code_vecs)
                    loss_intra = 0.1 * (loss_intra_nl_batch + loss_intra_code_batch)

                else:
                    # 只用简单的 Batch 内增强
                    loss_intra_code = hfedr_loss_two(code_vecs, aug_code_vecs)
                    loss_intra_nl = hfedr_loss_two(nl_vecs, aug_nl_vecs)
                    loss_intra = 0.1 * (loss_intra_code + loss_intra_nl)

                    loss_hncl = torch.tensor(0.0, device=args.device)
                    loss_intra_hard = torch.tensor(0.0, device=args.device)


                total_loss = loss_bcl + loss_hncl + loss_intra + loss_intra_hard
                # --- 累加 Loss 用于显示 ---
                tr_loss += total_loss.item()
                tr_loss_bcl += loss_bcl.item()
                tr_loss_hncl += loss_hncl.item()
                tr_loss_intra += loss_intra.item()
                tr_loss_intra_hard += loss_intra_hard.item()

                tr_num += 1

                if (step + 1) % 100 == 0 or step == 0:
                    avg_loss = tr_loss / tr_num
                    avg_bcl = tr_loss_bcl / tr_num
                    avg_hncl = tr_loss_hncl / tr_num
                    avg_intra = tr_loss_intra / tr_num
                    avg_intra_hard = tr_loss_intra_hard / tr_num

                    tqdm.write(
                        f"Ep {epoch + 1} st {step + 1} | "
                        f"Tot:{avg_loss:.4f} | BCL:{avg_bcl:.4f} | HNCL:{avg_hncl:.4f} |"
                        f"Intra:{avg_intra:.4f} | HN_Intra:{avg_intra_hard:.4f}"
                    )

                    # 重置统计变量
                    tr_loss, tr_loss_bcl, tr_loss_hncl, tr_loss_intra, tr_loss_intra_hard = 0.0, 0.0, 0.0, 0.0, 0.0
                    tr_num = 0

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # 每轮保存模型
        epoch_model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, epoch_model_path)
        print(f"Epoch {epoch+1} model saved to {epoch_model_path}")

        evaluating(args, model, tokenizer, epoch, pool)

        # 删除上一轮保存的模型
        if epoch > 0:
            previous_model_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
            if os.path.exists(previous_model_path):
                #os.remove(previous_model_path)
                #print(f"Removed previous model: {previous_model_path}")
                pass

def evaluating(args, model, tokenizer, epoch, pool):
    print("开始评估模型")
    model_path = f"outputs_model/model_epoch_{epoch+1}.pt"
    print(f"\n评估模型: {model_path}")
    # 只加载模型参数即可
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"Loaded model from {model_path}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(args, model, tokenizer, args.valid_data , pool)
        print(f"Evaluation Results for epoch {epoch+1}:")
        print(f"MRR   : {metrics['eval_mrr']:.4f}")
        print(f"Top1  : {metrics['top1']:.4f}")
        print(f"Top5  : {metrics['top5']:.4f}")
        print(f"Top10 : {metrics['top10']:.4f}")
        print(f"Top100: {metrics['top100']:.4f}")

def test_modle(args, model, tokenizer, pool):
    print("开始评估最好模型")
    model_path = f"outputs_model/model_epoch_.pt"
    print(f"\n评估模型: {model_path}")
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"Loaded model from {model_path}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(args, model, tokenizer, args.test_data, pool)
        print(f"Evaluation Results for best epoch:")
        print(f"MRR   : {metrics['eval_mrr']:.4f}")
        print(f"Top1  : {metrics['top1']:.4f}")
        print(f"Top5  : {metrics['top5']:.4f}")
        print(f"Top10 : {metrics['top10']:.4f}")
        print(f"Top100: {metrics['top100']:.4f}")

def evaluate_model(args, model, tokenizer, data_path, pool):
    model.eval()
    # 构造查询集（自然语言 + 上下文）
    query_dataset = TextDataset(tokenizer, args, data_path, pool=pool)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    # 构造代码集
    code_dataset = TextDatasetBase(tokenizer, args, args.codebase_file, pool=pool)
    code_dataloader = DataLoader(
        code_dataset,
        sampler=SequentialSampler(code_dataset),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    # 提取 query 向量
    nl_vecs = []
    for batch in tqdm(query_dataloader, desc="Encoding queries"):
        _, code_ids, nl_ids, package_ids, function_ids, levels = batch
        nl_inputs = nl_ids.to(args.device)
        package_inputs = package_ids.to(args.device)
        function_inputs = function_ids.to(args.device)
        with torch.no_grad(), amp.autocast("cuda"):
            nl_vec = model(nl_inputs = nl_inputs, context_inputs=[package_inputs, function_inputs], levels=levels)
            nl_vecs.append(nl_vec.detach())
    nl_vecs = torch.cat(nl_vecs, dim=0)

    # 提取 code 向量
    code_vecs = []
    for batch in tqdm(code_dataloader, desc="Encoding codebase"):
        code_ids, nl_ids = batch
        code_ids = code_ids.to(args.device)
        with torch.no_grad(), amp.autocast("cuda"):
            code_vec = model(code_inputs=code_ids)
            code_vecs.append(code_vec.detach())
    code_vecs = torch.cat(code_vecs, dim=0)

    model.train()

    # 提取 URL 用于匹配
    nl_urls = [example.url for example in query_dataset.examples]
    code_urls = [example.url for example in code_dataset.examples]

    # MRR & Top-k 计算
    ranks = []
    top1, top5, top10, top100 = 0, 0, 0, 0

    # 定义一个处理查询的批次大小，可以根据你的显存/内存调整
    batch_size = args.eval_batch_size

    for i in tqdm(range(0, len(nl_vecs), batch_size), desc="Calculating scores"):
        # 取出一个批次的 query 向量
        nl_batch_vecs = nl_vecs[i:i + batch_size]

        # 计算这个批次的 query 与整个 codebase 的分数
        # 使用 torch.matmul 在 GPU 上计算，速度更快
        scores = torch.matmul(nl_batch_vecs.half(), code_vecs.T.half())

        # 对分数进行排序，并移回 CPU 处理
        sort_ids = torch.argsort(scores, dim=-1, descending=True).cpu().numpy()

        # 获取这个批次对应的 URL
        batch_urls = nl_urls[i:i + batch_size]

        for url, sort_id in zip(batch_urls, sort_ids):
            rank = 0
            find = False
            # 只检查前 1000 个结果
            for idx in sort_id[:1000]:
                if not find:
                    rank += 1
                if code_urls[idx] == url:
                    find = True
                    break  # 找到后立即退出内层循环

            if find:
                ranks.append(1 / rank)
                if rank <= 1: top1 += 1
                if rank <= 5: top5 += 1
                if rank <= 10: top10 += 1
                if rank <= 100: top100 += 1
            else:
                ranks.append(0)

    results = {
        "eval_mrr": float(np.mean(ranks)),
        "top1": float(top1 / len(ranks)),
        "top5": float(top5 / len(ranks)),
        "top10": float(top10 / len(ranks)),
        "top100": float(top100 / len(ranks))
    }
    return results

def main(pool):
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 读参
    args = set_args()

    # 设置随机种子
    set_seed(args.seed)

    # 写训练日志
    write_arg_log(vars(args), args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # 设定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    gpu_num = torch.cuda.device_count()
    print(f"device: {device}, gpu_num: {gpu_num}")

    # 模型准备
    config = RobertaConfig.from_pretrained(args.pretrained_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)
    model = RobertaModel.from_pretrained(args.pretrained_model_path)
    model=Model(model)

    # 开始训练
    print("开始训练")
    # 将模型加载到指定的设备上
    model.to(device)

    # 调用train_model函数进行训练
    train_model(args, model, tokenizer, pool)
    # 测试
    # test_modle(args, model, tokenizer, pool)


if __name__ == "__main__":
    print("等待 GPU 空闲（空闲显存超过 {} MB）...".format(MEMORY_FREE_THRESHOLD_MB))
    while True:
        free_mem = get_gpu0_free_memory()
        msg = f"\r当前 GPU 0 空闲显存: {free_mem} MB"
        sys.stdout.write(msg)
        sys.stdout.flush()

        if free_mem >= MEMORY_FREE_THRESHOLD_MB:
            print("\n显卡空闲，开始执行主程序。")
            break
        time.sleep(60)
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    with multiprocessing.Pool(cpu_count) as pool:
        main(pool)