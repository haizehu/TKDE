import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    (保持不变)
    """

    def __init__(self, size, dim, is_index):
        super(MemoryBank, self).__init__()
        self.size = size
        self.is_index = is_index
        if is_index:
            self.register_buffer('bank', torch.full((1, size), -1, dtype=torch.long))
        else:
            init_vecs = torch.randn(dim, size)
            init_vecs = F.normalize(init_vecs, p=2, dim=0) * 1e-9
            self.register_buffer('bank', init_vecs)
        self.register_buffer('bank_ptr', torch.LongTensor([0]))

    @torch.no_grad()
    def get_bank(self):
        return self.bank.t()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        bs = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + bs >= self.size:
            tail_len = self.size - ptr
            self.bank[:, ptr:] = batch[:tail_len].T
            head_len = bs - tail_len
            if head_len > 0:
                self.bank[:, :head_len] = batch[tail_len:].T
            self.bank_ptr[0] = head_len
        else:
            self.bank[:, ptr:ptr + bs] = batch.T
            self.bank_ptr[0] = ptr + bs

    @torch.no_grad()
    def update(self, indexs, vecs):
        self.bank[:, indexs] = vecs.T

    @torch.no_grad()
    def update_all_features(self, new_bank_tensor):
        if new_bank_tensor.shape != self.bank.shape:
            if new_bank_tensor.T.shape == self.bank.shape:
                new_bank_tensor = new_bank_tensor.T
            else:
                raise ValueError(f"Shape mismatch: {self.bank.shape} vs {new_bank_tensor.shape}")
        self.bank.copy_(new_bank_tensor)


class NeighborBank(nn.Module):
    def __init__(self, size, dim, K=10):
        super(NeighborBank, self).__init__()
        self.queue = MemoryBank(size=size, dim=dim, is_index=False)
        self.K = K

    @torch.no_grad()
    def put(self, batch):
        self.queue._dequeue_and_enqueue(batch)

    @torch.no_grad()
    def get_neighbors_dynamic(self, batch, positive_indices, index_bank_tensor):
        B, H = batch.shape
        K = self.K
        device = batch.device

        # 计算所有样本相似度 [B, Bank_Size]
        bank = self.queue.get_bank()  # [Size, Dim]
        sim_mat = torch.einsum('nd,md->nm', batch, bank)

        # 定位正样本
        # pos_mask: [B, Size] (True 表示是正样本)
        pos_mask = (positive_indices.unsqueeze(1) == index_bank_tensor.unsqueeze(0))

        # 提取正样本的分数
        pos_loc_idx = pos_mask.max(dim=1)[1]  # [B]
        # 提取分数 [B, 1]
        pos_scores = sim_mat.gather(1, pos_loc_idx.unsqueeze(1))

        # 屏蔽正样本
        sim_mat.masked_fill_(pos_mask, -float('inf'))

        # 对负样本进行全局排序
        sorted_scores, sorted_indices = torch.sort(sim_mat, dim=1, descending=True)

        # 计算正样本的“虚拟排名”
        # harder_mask: [B, Size]
        harder_mask = sorted_scores > pos_scores
        ranks = harder_mask.sum(dim=1)  # [B]，每个样本对应的“比它难的负样本数”

        # 确定截取窗口

        start_indices = (ranks - K).clamp(min=0)  # [B]

        # 构造 Gather 索引矩阵
        # [0, 1, 2, ..., K-1]
        offsets = torch.arange(K, device=device).unsqueeze(0)  # [1, K]
        # [B, K] -> 每一行都是 [start, start+1, ..., start+K-1]
        gather_indices = start_indices.unsqueeze(1) + offsets

        # 防止索引越界 (虽然理论上 Bank 很大 K 很小不会越界，加个保险)
        max_idx = self.queue.size - 1
        gather_indices = gather_indices.clamp(max=max_idx)

        # 提取最终的硬负样本索引
        final_indices = torch.gather(sorted_indices, 1, gather_indices)

        return final_indices

    @torch.no_grad()
    def update(self, indexs, vecs):
        self.queue.update(indexs=indexs, vecs=vecs)

    @torch.no_grad()
    def update_all(self, new_vecs):
        self.queue.update_all_features(new_vecs)


class VecIndexBank(nn.Module):

    def __init__(self, size, dim, K=10):
        super(VecIndexBank, self).__init__()
        self.size = size
        self.K = K
        self.vec_bank = NeighborBank(size=size, dim=dim, K=K)
        self.index_bank = MemoryBank(size=size, dim=1, is_index=True)

    @torch.no_grad()
    def get_index(self, batch, positive_indices):
        index_bank_tensor = self.index_bank.get_bank().squeeze()
        bank_pos_idxs = self.vec_bank.get_neighbors_dynamic(
            batch=batch,
            positive_indices=positive_indices,
            index_bank_tensor=index_bank_tensor
        )
        # 获取 dataset 中的真实索引 (Data ID) 用于后续提取原始文本
        flat_bank_pos_idxs = bank_pos_idxs.flatten()
        data_idxs = torch.index_select(index_bank_tensor, dim=0, index=flat_bank_pos_idxs).long()
        return data_idxs, flat_bank_pos_idxs

    @torch.no_grad()
    def update(self, indexs, vecs):
        self.vec_bank.update(indexs=indexs, vecs=vecs)

    @torch.no_grad()
    def put(self, batch, indexs):
        self.vec_bank.put(batch)
        self.index_bank._dequeue_and_enqueue(indexs.unsqueeze(1))

    @torch.no_grad()
    def apply_axbn_update(self, new_vecs):
        self.vec_bank.update_all(new_vecs)