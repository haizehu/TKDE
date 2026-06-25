import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData, Batch

class GatingFusion(nn.Module):

    def __init__(self, hidden_size):
        super(GatingFusion, self).__init__()
        self.hidden_size = hidden_size

        # 定义用于计算门控信号的 MLP
        # 输入是两个向量的拼接 (2 * hidden_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),  # 使用平滑的激活函数增加非线性
            nn.Sigmoid()  # Sigmoid 函数确保输出值在 (0, 1) 区间，作为门控信号
        )

        # 添加一个输出MLP，对融合后的向量做进一步的非线性变换
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, nl_vec, context_embs):
        # 将两个向量拼接，作为计算门控信号的输入
        combined_vec = torch.cat([nl_vec, context_embs], dim=1)

        # 通过 MLP 计算门控信号 gate
        gate = self.gate_mlp(combined_vec)

        # 应用门控机制进行加权融合
        gated_fused_vec = gate * nl_vec + (1 - gate) * context_embs

        # 将融合后的信息作为对原始 nl_vec 的一个增量更新
        output_vec = self.output_mlp(gated_fused_vec)
        final_vec = self.layer_norm(gated_fused_vec + output_vec)

        return final_vec

class ContextGNNEncoderHetero(nn.Module):
    def __init__(self, hidden_size=768, gat_heads=4, gat_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads

        self.in_linear_dict = nn.ModuleDict({
            'query': nn.Linear(hidden_size, hidden_size),
            'package': nn.Linear(hidden_size, hidden_size),
            'function': nn.Linear(hidden_size, hidden_size),
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gat_layers):
            conv = HeteroConv({
                # 局部调用关系
                ('function', 'calls', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                             add_self_loops=False),
                ('function', 'is_called_by', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),

                # 全局枢纽关系
                ('function', 'gathers_from', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),
                ('function', 'broadcasts_to', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                     add_self_loops=False),


                ('package', 'in_context_of', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),
                ('query', 'attends_to', 'package'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                              add_self_loops=False),
                ('query', 'attends_to', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                               add_self_loops=False),
                ('function', 'context_for', 'package'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                  add_self_loops=False),
                ('package', 'attended_by', 'query'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                               add_self_loops=False),
                ('function', 'attended_by', 'query'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

            # 为每种节点类型都创建一个 LayerNorm 实例
            norm_dict = nn.ModuleDict({
                'query': nn.LayerNorm(hidden_size),
                'package': nn.LayerNorm(hidden_size),
                'function': nn.LayerNorm(hidden_size),
            })
            self.norms.append(norm_dict)
        self.output_mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, batched_data: Batch):
        """
        直接处理一个批量化的图对象 (Batch object)。
        """
        x_dict = batched_data.x_dict
        edge_index_dict = batched_data.edge_index_dict

        edge_index_dict_to_use = edge_index_dict

        # --- GNN 聚合 ---
        # 应用初始线性变换和激活函数
        for node_type, x in x_dict.items():
            # x_dict[node_type] = self.in_linear_dict[node_type](x).relu()
            x_dict[node_type] = F.gelu(self.in_linear_dict[node_type](x))

        # 执行多层GNN卷积
        for i, conv in enumerate(self.convs):
            x_dict_input = x_dict

            x_dict_updates = conv(x_dict_input, edge_index_dict_to_use)

            x_dict_next = {}

            for node_type in x_dict_input.keys():
                x_original = x_dict_input[node_type]

                if node_type in x_dict_updates:
                    x_update = x_dict_updates[node_type]
                else:
                    # 如果是孤立节点，它的消息更新量就是 0
                    x_update = torch.zeros_like(x_original)

                # 残差连接
                x_res = x_original + x_update

                # 归一化
                x_norm = self.norms[i][node_type](x_res)

                # 激活
                x_activated = F.gelu(x_norm)

                # 存入新字典
                x_dict_next[node_type] = x_activated

            # 将更新后的特征全集赋给 x_dict，准备进入下一层 GNN
            x_dict = x_dict_next

        # --- 获取最终输出 ---
        query_final_emb = x_dict['query']
        context_vec = self.output_mlp(query_final_emb)

        return context_vec

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embeddings = encoder.embeddings

        # 获取隐藏层大小
        hidden_size = 768

        self.context_gnn_encoder = ContextGNNEncoderHetero(hidden_size=768, gat_heads=4, gat_layers=2)

        self.FusionModule = GatingFusion(hidden_size=hidden_size)

        self.code_residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        self.nl_residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, code_inputs=None, nl_inputs=None, context_inputs=None, levels=None):
        if code_inputs is not None:
            output = self.encoder(code_inputs,
                                  attention_mask=code_inputs.ne(1),
                                  output_hidden_states=True)

            output_org = output[0]  # 形状: [B, L, H]
            mask_code = code_inputs.ne(1)[:, :, None]
            # 沿序列长度维度求和后除以有效 Token 数量，形状变为: [B, H]
            output_org = (output_org * mask_code).sum(1) / code_inputs.ne(1).sum(-1)[:, None]

            return F.normalize(output_org, p=2, dim=1)

        elif nl_inputs is not None:
            output = self.encoder(nl_inputs,
                                  attention_mask=nl_inputs.ne(1),
                                  output_hidden_states=True)

            # 对最后一层做 Masked Mean Pooling
            output_org = output[0]  # 形状: [B, L, H]
            mask_nl = nl_inputs.ne(1)[:, :, None]
            # 沿序列长度维度求和后除以有效 Token 数量，形状变为: [B, H]
            output_org = (output_org * mask_nl).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]

            output_org = F.normalize(output_org, p=2, dim=1)
            return self._process_context_batched(output_org, context_inputs, levels)

    def _process_context_batched(self, nl_vec, context_inputs, levels):
        package_inputs, function_inputs = context_inputs
        B = package_inputs.shape[0]  # Batch size
        P = package_inputs.shape[1]  # max_packages
        num_funcs = function_inputs.shape[1]  # max_functions (修复变量名冲突)
        device = nl_vec.device

        with torch.no_grad():
            # ================= 1. 提取动态特征 =================
            all_package_inputs = package_inputs.view(-1, package_inputs.size(-1))
            all_function_inputs = function_inputs.view(-1, function_inputs.size(-1))

            package_outputs = self.encoder(all_package_inputs, attention_mask=all_package_inputs.ne(1),
                                           output_hidden_states=True)[0]
            # package_encodings_flat = self.encode_layerwise(package_outputs, inputs=all_package_inputs,
            #                                                encoder_type='code_dynamic_selection')
            pkg_valid_lens = all_package_inputs.ne(1).sum(-1).clamp(min=1e-9)[:, None]
            package_encodings_flat = (package_outputs * all_package_inputs.ne(1)[:, :, None]).sum(1) / pkg_valid_lens

            function_outputs = self.encoder(all_function_inputs, attention_mask=all_function_inputs.ne(1),
                                            output_hidden_states=True)[0]
            # function_encodings_flat = self.encode_layerwise(function_outputs, inputs=all_function_inputs,
            #                                                 encoder_type='code_dynamic_selection')
            func_valid_lens = all_function_inputs.ne(1).sum(-1).clamp(min=1e-9)[:, None]
            function_encodings_flat = (function_outputs * all_function_inputs.ne(1)[:, :, None]).sum(
                1) / func_valid_lens

            # 归一化
            package_encodings_flat = F.normalize(package_encodings_flat, p=2, dim=-1)
            function_encodings_flat = F.normalize(function_encodings_flat, p=2, dim=-1)

            package_encodings = package_encodings_flat.view(B, P, -1)

            # ================= 向量化构建大图 =================
            batched_data = HeteroData()


            # Query 节点: B 个
            batched_data['query'].x = nl_vec

            # Function 节点: B * num_funcs 个 (直接铺平)
            batched_data['function'].x = function_encodings_flat

            # Package 节点: 动态过滤占位符
            valid_pkg_mask = package_inputs[:, :, 0] != 1  # [B, P]
            batched_data['package'].x = package_encodings[valid_pkg_mask]  # [V, H], V是全局有效package总数

            # --- 全局索引映射计算 ---
            pkg_batch_idx, _ = valid_pkg_mask.nonzero(as_tuple=True)  # [V]
            V = pkg_batch_idx.size(0)

            # --- 边构建---

            # 关系 1 & 3: Query <--> Package
            if V > 0:
                p_indices = torch.arange(V, device=device)
                edge_index_q_p = torch.stack([pkg_batch_idx, p_indices], dim=0)
                batched_data['query', 'attends_to', 'package'].edge_index = edge_index_q_p
                batched_data['package', 'attended_by', 'query'].edge_index = edge_index_q_p[[1, 0]]
            else:
                batched_data['query', 'attends_to', 'package'].edge_index = torch.empty((2, 0), dtype=torch.long,
                                                                                        device=device)
                batched_data['package', 'attended_by', 'query'].edge_index = torch.empty((2, 0), dtype=torch.long,
                                                                                         device=device)

            # 关系 2: Package <--> Function
            if V > 0 and num_funcs > 0:
                # 每个有效 package 连接同批次的所有 num_funcs 个 function
                p_idx_repeat = torch.arange(V, device=device).repeat_interleave(num_funcs)  # [V * num_funcs]
                # 计算对应的 Function 全局索引
                f_idx_base = (pkg_batch_idx * num_funcs).unsqueeze(1) + torch.arange(num_funcs,
                                                                                     device=device).unsqueeze(
                    0)  # [V, num_funcs]
                edge_index_p_f = torch.stack([p_idx_repeat, f_idx_base.flatten()], dim=0)

                batched_data['package', 'in_context_of', 'function'].edge_index = edge_index_p_f
                batched_data['function', 'context_for', 'package'].edge_index = edge_index_p_f[[1, 0]]
            else:
                batched_data['package', 'in_context_of', 'function'].edge_index = torch.empty((2, 0), dtype=torch.long,
                                                                                              device=device)
                batched_data['function', 'context_for', 'package'].edge_index = torch.empty((2, 0), dtype=torch.long,
                                                                                            device=device)

            # 关系 4: Query <--> Function
            if num_funcs > 0:
                q_indices_f = torch.arange(B * num_funcs, device=device) // num_funcs
                f_indices_all = torch.arange(B * num_funcs, device=device)
                edge_index_q_f = torch.stack([q_indices_f, f_indices_all], dim=0)

                batched_data['query', 'attends_to', 'function'].edge_index = edge_index_q_f
                batched_data['function', 'attended_by', 'query'].edge_index = edge_index_q_f[[1, 0]]

            # 关系 5: Function <--> Function (调用 & 枢纽)
            levels_tensor = torch.as_tensor(levels, device=device)  # [B, num_funcs]
            real_mask = (levels_tensor != -1)
            pad_mask = (levels_tensor == -1)

            # 5.1 局部调用边
            call_adj = (levels_tensor.unsqueeze(2) < levels_tensor.unsqueeze(1)) & real_mask.unsqueeze(
                2) & real_mask.unsqueeze(1)
            b_call, src_local, dst_local = call_adj.nonzero(as_tuple=True)
            src_global = b_call * num_funcs + src_local
            dst_global = b_call * num_funcs + dst_local
            edge_index_calls = torch.stack([src_global, dst_global], dim=0)

            batched_data['function', 'calls', 'function'].edge_index = edge_index_calls
            batched_data['function', 'is_called_by', 'function'].edge_index = edge_index_calls[[1, 0]]

            # 5.2 全局枢纽边 (Global Hubs)
            hub_adj = real_mask.unsqueeze(2) & pad_mask.unsqueeze(1)  # [B, num_funcs, num_funcs]
            b_hub, real_local, pad_local = hub_adj.nonzero(as_tuple=True)
            real_global = b_hub * num_funcs + real_local
            pad_global = b_hub * num_funcs + pad_local
            edge_index_hub = torch.stack([real_global, pad_global], dim=0)

            batched_data['function', 'gathers_from', 'function'].edge_index = edge_index_hub
            batched_data['function', 'broadcasts_to', 'function'].edge_index = edge_index_hub[[1, 0]]

        # ================= 3. 传入 GNN 进行计算 =================
        context_embs = self.context_gnn_encoder(batched_data)

        # 融合
        fused_vec = self.FusionModule(nl_vec, context_embs)
        return F.normalize(fused_vec, p=2, dim=-1)