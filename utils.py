import sys

max_functions = 30
max_packages = 30
reserved_info_slots = 5 # 保留信息位的数量

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 package_ids,
                 function_ids,
                 levels,
                 up_fun_num

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.package_ids = package_ids
        self.function_ids = function_ids
        self.levels = levels
        self.up_fun_num = up_fun_num
class InputFeaturesBase(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

def findIndex(levels):
    return levels[:max_functions] + [-1] * (max_functions - len(levels))


def convert_examples_to_features(item):
    js, tokenizer, args = item
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    package_tokens = []
    package_ids = []
    for pack in js['package']:
        pack_token = tokenizer.tokenize(pack)[:args.context_length - 2]
        pack_token = [tokenizer.cls_token] + pack_token + [tokenizer.sep_token]
        pack_id = tokenizer.convert_tokens_to_ids(pack_token)
        padding_length = args.context_length - len(pack_id)
        pack_id += [tokenizer.pad_token_id] * padding_length
        package_tokens.append(pack_token)
        package_ids.append(pack_id)
    package_ids = package_ids[:max_packages]
    package_ids += [[tokenizer.pad_token_id] * args.context_length] * (max_packages - len(package_ids))

    # ==================== 寻找最优结构起点 ====================
    original_functions = js['function']
    original_levels = js['levels']
    up_fun_num = max(0, min(js['up_fun_num'] - 1, len(original_levels) - 1))  # 同时确保它不是负数

    if len(original_levels) > max_functions:
        # 定义候选区域
        search_start = max(0, up_fun_num - max_functions + 1)
        search_end = up_fun_num + 1

        # 在候选区域内，寻找 level 最小的索引
        best_start_index = up_fun_num
        min_level_found = original_levels[up_fun_num] if up_fun_num < len(original_levels) else sys.maxsize

        # 从后往前遍历候选区域，寻找更好的（level更小）起点
        for i in range(search_end - 2, search_start - 1, -1):
            if original_levels[i] <= min_level_found:
                min_level_found = original_levels[i]
                best_start_index = i

        # 确定最终的窗口
        start_index = best_start_index
        end_index = start_index + max_functions

        processed_functions = original_functions[start_index:end_index]
        processed_levels = original_levels[start_index:end_index]
    else:
        # 如果未超出，则直接使用原始列表
        processed_functions = original_functions
        processed_levels = original_levels
    # ================================================================

    # 在处理过的列表上进行后续操作
    levels = findIndex(processed_levels)
    # 额外加入5个-1作为信息位
    levels += [-1] * reserved_info_slots

    function_tokens = []
    function_ids = []
    if processed_functions:
        for func in processed_functions:
            func_token = tokenizer.tokenize(func)[:args.context_length - 2]
            func_token = [tokenizer.cls_token] + func_token + [tokenizer.sep_token]
            func_id = tokenizer.convert_tokens_to_ids(func_token)
            padding_length = args.context_length - len(func_id)
            func_id += [tokenizer.pad_token_id] * padding_length
            function_tokens.append(func_token)
            function_ids.append(func_id)

    # 填充 function_ids 至固定长度
    function_ids += [[tokenizer.pad_token_id] * args.context_length] * (max_functions - len(function_ids))
    # 2. 额外加入5个虚拟函数作为固定填充
    function_ids += [[tokenizer.pad_token_id] * args.context_length] * reserved_info_slots

    up_fun_num = js['up_fun_num']

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids, js['url'] if "url" in js else js["retrieval_idx"], package_ids, function_ids,
                         levels, up_fun_num)

def convert_examples_to_features_base(item):
    js, tokenizer, args = item
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeaturesBase(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])
