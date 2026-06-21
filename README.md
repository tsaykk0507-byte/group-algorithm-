```python
import re
import math
import random

# 設定隨機對照組生成組數 (B)
GAP_B_SAMPLES = 15

# ==============================================================================
# 1. 基礎統計與數學工具
# ==============================================================================
def flatten_list(nested_list):
    flat = []
    for item in nested_list:
        if isinstance(item, list): flat.extend(flatten_list(item))
        else: flat.append(float(item))
    return flat

def parse_input_string(input_str):
    clean_str = input_str.replace('，', ',').replace(' ', '').replace('\t', '')
    parts = clean_str.split(',')
    data_pool = {}
    for p in parts:
        if not p: continue
        p_standard = p.replace('×', 'x').replace('X', 'x').replace('*', 'x')
        if 'x' in p_standard:
            match = re.match(r'^(\d+)[xX*]([+-]?\d*\.?\d+)$', p_standard)
            if match:
                val = float(match.group(2))
                data_pool[val] = data_pool.get(val, 0) + int(match.group(1))
        else:
            try:
                val = float(p)
                data_pool[val] = data_pool.get(val, 0) + 1
            except ValueError: continue
    flat_data = []
    for val, count in data_pool.items(): flat_data.extend([val] * count)
    flat_data.sort()
    return flat_data

def get_median(lst):
    if not lst: return 0.0
    sl = sorted(lst)
    n = len(sl)
    mid = n // 2
    return (sl[mid-1] + sl[mid]) / 2.0 if n % 2 == 0 else float(sl[mid])

# 👑 修正版全域 Gap Statistic 計算工具（改為對稱自我評估，不再獨厚 K-Means）
def calculate_gap_statistic(algorithm_func, clusters, flat_data, B=GAP_B_SAMPLES):
    k = len(clusters)
    n = len(flat_data)
    if k == 0 or n <= 1: return 0.0
        
    wk_actual = 0.0
    for c in clusters:
        if not c['elements']: continue
        mu = sum(c['elements']) / len(c['elements'])
        wk_actual += sum((x - mu) ** 2 for x in c['elements'])
        
    log_wk_actual = math.log(wk_actual + 1e-10)
    min_val, max_val = min(flat_data), max(flat_data)
    if min_val == max_val: return 0.0
        
    log_wk_refs = []
    for _ in range(B):
        ref_data = sorted([random.uniform(min_val, max_val) for _ in range(n)])
        # 🎯 修正點：對照組使用同一個演算法進行分群評估，確保基準公平
        try:
            ref_clusters, _ = algorithm_func(ref_data)
            # 轉換格式以符合 W_k 計算需求
            wk_ref = 0.0
            for rc in ref_clusters:
                if not rc['elements']: continue
                mu_ref = sum(rc['elements']) / len(rc['elements'])
                wk_ref += sum((rx - mu_ref) ** 2 for rx in rc['elements'])
            log_wk_refs.append(math.log(wk_ref + 1e-10))
        except:
            log_wk_refs.append(log_wk_actual)
            
    expected_log_wk = sum(log_wk_refs) / len(log_wk_refs) if log_wk_refs else log_wk_actual
    return max(0.0, expected_log_wk - log_wk_actual)

# ==============================================================================
# 2. 傳統 K-Means (修正初始化與空群 Bug)
# ==============================================================================
def run_kmeans_core(flat_data, k):
    n = len(flat_data)
    if k >= n: return [[x] for x in flat_data], flat_data[:]
    
    # 🎯 修正點：改用分位數（Quantiles）初始化，避免等距初始化導致質心偏離或大量空群
    flat_data_sorted = sorted(flat_data)
    centroids = [flat_data_sorted[int(i * (n - 1) / (k - 1))] for i in range(k)] if k > 1 else [sum(flat_data)/n]
    
    for _ in range(50):
        clusters = [[] for _ in range(k)]
        for x in flat_data:
            distances = [abs(x - c) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(x)
            
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroids.append(sum(cluster) / len(cluster))
            else:
                # 🎯 修正點：空群時隨機挑選一個當前數據點作為新質心，打破僵局
                new_centroids.append(random.choice(flat_data))
        if sorted(centroids) == sorted(new_centroids): break
        centroids = new_centroids
    return clusters, centroids

def optimized_kmeans(flat_data):
    if len(set(flat_data)) < 3: k_range = [2]
    else: k_range = list(range(2, min(7, len(set(flat_data)) + 1)))
    best_k = k_range[0]
    best_score = -2.0
    best_clusters = []
    
    for k in k_range:
        clusters, _ = run_kmeans_core(flat_data, k)
        scores = []
        for c_idx, clus in enumerate(clusters):
            if not clus: continue
            for x in clus:
                a_i = sum(abs(x - y) for y in clus) / (len(clus) - 1) if len(clus) > 1 else 0.0
                other_dists = []
                for o_idx, other in enumerate(clusters):
                    if o_idx != c_idx and other:
                        other_dists.append(sum(abs(x - y) for y in other) / len(other))
                b_i = min(other_dists) if other_dists else 0.0
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
                scores.append(s_i)
        avg_score = sum(scores) / len(scores) if scores else -1.0
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 3. 傳統 GMM 高斯混合 (修正機率崩潰與數值穩定性 Bug)
# ==============================================================================
def pdf(x, mu, var):
    if var < 1e-6: var = 1e-6  # 🎯 防禦方差過小
    try:
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mu)**2) / (2 * var))
    except OverflowError:
        return 0.0

def run_gmm_core(flat_data, k):
    n = len(flat_data)
    global_mean = sum(flat_data) / n
    global_var = sum((x - global_mean)**2 for x in flat_data) / n
    if global_var == 0: global_var = 1.0
    
    flat_data_sorted = sorted(flat_data)
    means = [flat_data_sorted[int(i * (n - 1) / (k - 1))] for i in range(k)] if k > 1 else [global_mean]
    vars_ = [global_var] * k
    weights = [1.0 / k] * k
    resp = [[0.0] * k for _ in range(n)]
    
    for _ in range(30):
        # E-step
        for i in range(n):
            total_density = 0.0
            for j in range(k):
                resp[i][j] = weights[j] * pdf(flat_data[i], means[j], vars_[j])
                total_density += resp[i][j]
            if total_density < 1e-12:
                # 🎯 修正點：若完全脫離所有成分，均勻平分而非產生極端權重震盪
                resp[i] = [1.0 / k] * k
            else:
                for j in range(k): resp[i][j] /= total_density
        # M-step
        for j in range(k):
            N_j = sum(resp[i][j] for i in range(n))
            if N_j < 1e-4: N_j = 1e-4
            means[j] = sum(resp[i][j] * flat_data[i] for i in range(n)) / N_j
            vars_[j] = sum(resp[i][j] * (flat_data[i] - means[j])**2 for i in range(n)) / N_j
            if vars_[j] < 1e-4: vars_[j] = 1e-4  # 🎯 方差下限保護
            weights[j] = N_j / n
            
    log_likelihood = 0.0
    for i in range(n):
        sample_density = sum(weights[j] * pdf(flat_data[i], means[j], vars_[j]) for j in range(k))
        if sample_density > 0: log_likelihood += math.log(sample_density)
    return resp, means, log_likelihood

def optimized_gmm(flat_data):
    n = len(flat_data)
    if len(set(flat_data)) < 3: k_range = [2]
    else: k_range = list(range(2, min(7, len(set(flat_data)) + 1)))
    best_k = k_range[0]
    min_bic = float('inf')
    best_resp = []
    
    for k in k_range:
        resp, _, log_likelihood = run_gmm_core(flat_data, k)
        num_params = 3 * k - 1
        bic = num_params * math.log(n) - 2 * log_likelihood
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_resp = resp
            
    clusters = [[] for _ in range(best_k)]
    for i in range(n):
        max_idx = best_resp[i].index(max(best_resp[i]))
        clusters[max_idx].append(flat_data[i])
        
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 4. 傳統 DBSCAN (修正核心點擴展與重複歸類 Bug)
# ==============================================================================
def optimized_dbscan(flat_data):
    if not flat_data: return [], 0.0
    n = len(flat_data)
    if n < 2: return [{"value": flat_data[0], "elements": flat_data}], 0.0
    
    dists = [abs(flat_data[i+1] - flat_data[i]) for i in range(n - 1)]
    eps = sum(dists) / len(dists) * 0.5
    if eps == 0: eps = 1e-4
    
    labels = [-1] * n  # -1 表示未分類/雜訊
    cluster_id = 0
    
    # 🎯 修正點：嚴格按照標準 DBSCAN 兩階段核心點判定與蔓延邏輯
    for i in range(n):
        if labels[i] != -1: continue
        neighbors = [j for j in range(n) if abs(flat_data[i] - flat_data[j]) <= eps]
        
        if len(neighbors) < 2:  # MinPts = 2
            continue
            
        labels[i] = cluster_id
        queue = [idx for idx in neighbors if idx != i]
        
        while queue:
            current_idx = queue.pop(0)
            if labels[current_idx] == -1:  # 雜訊變邊界點
                labels[current_idx] = cluster_id
            elif labels[current_idx] != -1:
                continue
                
            next_neighbors = [j for j in range(n) if abs(flat_data[current_idx] - flat_data[j]) <= eps]
            if len(next_neighbors) >= 2:
                for nn in next_neighbors:
                    if labels[nn] == -1 and nn not in queue:
                        queue.append(nn)
        cluster_id += 1
        
    cluster_map = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map: cluster_map[label] = []
        cluster_map[label].append(flat_data[idx])
        
    result = []
    for lid, elements in cluster_map.items():
        if lid == -1:
            # 雜訊點各成一獨立核心群
            for e in elements: result.append({"value": e, "elements": [e]})
        else:
            result.append({"value": sum(elements) / len(elements), "elements": sorted(elements)})
    return sorted(result, key=lambda x: x['value']), eps

# ==============================================================================
# 5. 傳統 Mean Shift (修正為標準高斯核)
# ==============================================================================
def run_mean_shift(flat_data):
    if not flat_data: return [], 0.0
    n = len(flat_data)
    if n < 2: return [{"value": flat_data[0], "elements": flat_data}], 0.0
    
    dists = [abs(flat_data[i+1] - flat_data[i]) for i in range(n - 1)]
    bandwidth = sum(dists) / len(dists) * 1.5 
    if bandwidth == 0: bandwidth = 1e-4
    
    shifted_points = []
    for x in flat_data:
        current = x
        for _ in range(30):
            # 🎯 修正點：改用標準高斯核權重代替 Flat Kernel，保證平滑收斂
            weights = []
            total_w = 0.0
            num_sum = 0.0
            for p in flat_data:
                dist_sq = (p - current) ** 2
                w = math.exp(-dist_sq / (2 * (bandwidth ** 2)))
                num_sum += p * w
                total_w += w
            new_mean = num_sum / total_w if total_w > 0 else current
            if abs(new_mean - current) < 1e-4: break
            current = new_mean
        shifted_points.append(round(current, 3))
        
    unique_peaks = sorted(list(set(shifted_points)))
    clusters_dict = {}
    for pt, peak in zip(flat_data, shifted_points):
        # 將漂移到相同峰值的點歸為同群
        closest_peak = min(unique_peaks, key=lambda p: abs(p - peak))
        if closest_peak not in clusters_dict: clusters_dict[closest_peak] = []
        clusters_dict[closest_peak].append(pt)
        
    result = [{"value": sum(clus) / len(clus), "elements": sorted(clus)} for clus in clusters_dict.values() if clus]
    return sorted(result, key=lambda x: x['value']), bandwidth

# ==============================================================================
# 6. 傳統 AGNES 階層分群 (修正孤立點輪廓高估 Bug)
# ==============================================================================
def run_agnes_core(flat_data, k):
    clusters = [[x] for x in flat_data]
    while len(clusters) > k:
        min_dist = float('inf')
        merge_pair = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = sum(abs(x - y) for x in clusters[i] for y in clusters[j]) / (len(clusters[i]) * len(clusters[j]))
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    return clusters

def optimized_agnes(flat_data):
    if len(set(flat_data)) < 3: k_range = [2]
    else: k_range = list(range(2, min(7, len(set(flat_data)) + 1)))
    best_k = k_range[0]
    best_score = -2.0
    best_clusters = []
    
    for k in k_range:
        clusters = run_agnes_core(flat_data, k)
        scores = []
        for c_idx, clus in enumerate(clusters):
            for x in clus:
                # 🎯 修正點：孤立點（len==1）時，a_i 應視為 0，且不能跳過評估以免高估整體輪廓
                a_i = sum(abs(x - y) for y in clus) / (len(clus) - 1) if len(clus) > 1 else 0.0
                other_dists = []
                for o_idx, other in enumerate(clusters):
                    if o_idx != c_idx and other:
                        other_dists.append(sum(abs(x - y) for y in other) / len(other))
                b_i = min(other_dists) if other_dists else 0.0
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
                scores.append(s_i)
        avg_score = sum(scores) / len(scores) if scores else -1.0
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 7. 一維專用 Jenks 自然裂點法 (修正重合數據邊界與索引錯位 Bug)
# ==============================================================================
def jenks_breaks(data, k):
    data = sorted(data)
    n = len(data)
    mat1 = [[0] * (k + 1) for _ in range(n + 1)]
    mat2 = [[0.0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, k + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0.0
        for j in range(2, n + 1): mat2[j][i] = float('inf')
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = data[i3 - 1]
            s1 += val
            s2 += val * val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    # 🎯 修正點：嚴格限定大於，並加入 1e-9 擾動防止相同重複數據造成斷點死鎖
                    if mat2[l][j] > (v + mat2[i4][j - 1] + 1e-9):
                        mat2[l][j] = v + mat2[i4][j - 1]
                        mat1[l][j] = i3
        mat1[l][1] = 1
        mat2[l][1] = v
    kclass = [0] * (k + 1)
    kclass[k] = data[n - 1]
    kclass[0] = data[0]
    count_num = k
    backtrack = n
    while count_num >= 2:
        id_idx = int(mat1[backtrack][count_num]) - 1
        if id_idx < 0: id_idx = 0
        kclass[count_num - 1] = data[id_idx]
        backtrack = id_idx
        count_num -= 1
    return kclass

def optimized_jenks(flat_data):
    if len(set(flat_data)) < 3: k_range = [2]
    else: k_range = list(range(2, min(7, len(set(flat_data)) + 1)))
    best_k = k_range[0]
    best_gvf = -1.0
    best_clusters = []
    data_mean = sum(flat_data) / len(flat_data)
    sdam = sum((x - data_mean)**2 for x in flat_data)
    
    for k in k_range:
        try:
            breaks = jenks_breaks(flat_data, k)
            clusters = [[] for _ in range(k)]
            for x in flat_data:
                assigned = False
                for i in range(k):
                    if x <= breaks[i+1]: clusters[i].append(x); assigned = True; break
                if not assigned: clusters[-1].append(x)
            sdcm = 0.0
            for clus in clusters:
                if not clus: continue
                c_mean = sum(clus) / len(clus)
                sdcm += sum((x - c_mean)**2 for x in clus)
            gvf = (sdam - sdcm) / sdam if sdam != 0 else 1.0
            if gvf > best_gvf:
                best_gvf = gvf
                best_k = k
                best_clusters = clusters
        except: continue
        
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 8. 一維最優 K-Means (修正 L1/L2 度量衝突 Bug)
# ==============================================================================
def ckmeans_1d_dp(flat_data):
    n = len(flat_data)
    if n == 0: return [], 0
    if n == 1: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sub = flat_data[i:j+1]
            m = sum(sub) / len(sub)
            dist_matrix[i][j] = sum((x - m)**2 for x in sub)
            
    if len(set(flat_data)) < 3: k_range = [2]
    else: k_range = list(range(2, min(7, len(set(flat_data)) + 1)))
    
    best_k = k_range[0]
    best_clusters = []
    best_score = float('inf') # 🎯 改為最小化全域平方誤差和
    
    for k in k_range:
        if k > n: continue
        dp = [[float('inf')] * (n + 1) for _ in range(k + 1)]
        tb = [[0] * (n + 1) for _ in range(k + 1)]
        for j in range(1, n + 1): dp[1][j] = dist_matrix[0][j-1]
            
        for i in range(2, k + 1):
            for j in range(i, n + 1):
                for m in range(i - 1, j):
                    cost = dp[i-1][m] + dist_matrix[m][j-1]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        tb[i][j] = m
        breaks = [n]
        curr = n
        for i in range(k, 1, -1):
            curr = tb[i][curr]
            breaks.append(curr)
        if breaks[-1] != 0: breaks.append(0)
        breaks.reverse()
        
        clusters = []
        for idx in range(len(breaks) - 1):
            start_pos = breaks[idx]
            end_pos = breaks[idx+1]
            if start_pos < end_pos: clusters.append(flat_data[start_pos:end_pos])
            
        # 🎯 修正點：外層尋參指標改為最小化平方變異和（與內部 L2 度量保持高度一致）
        total_ss = dp[k][n]
        if total_ss < best_score:
            best_score = total_ss
            best_k = k
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 9. Bayesian Blocks 貝氏區塊法 (修正重複數值 w=0 崩潰 Bug)
# ==============================================================================
def bayesian_blocks(flat_data):
    n = len(flat_data)
    if n <= 1: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    # 🎯 修正點：對一維連續區間邊緣做微幅微擾，徹底根絕 w=0 的數學邊界崩潰
    edges = [flat_data[0] - 1e-5]
    for i in range(n - 1):
        edges.append((flat_data[i] + flat_data[i+1]) / 2.0)
    edges.append(flat_data[-1] + 1e-5)
    block_length = [edges[i+1] - edges[i] for i in range(n)]
    
    p0 = 0.05
    ncp_prior = 4 - math.log(p0 / (min(n, 100) * 0.01 + 1e-5))
    best = [0.0] * (n + 1)
    last = [0] * (n + 1)
    
    for r in range(1, n + 1):
        max_fitness = -float('inf')
        best_k = 0
        for l in range(1, r + 1):
            w = sum(block_length[l-1:r])
            cnt = r - l + 1
            fitness = cnt * (math.log(cnt) - math.log(w if w > 0 else 1e-10)) - ncp_prior
            total_fitness = fitness + (best[l-1] if l > 1 else 0.0)
            if total_fitness > max_fitness:
                max_fitness = total_fitness
                best_k = l - 1
        best[r] = max_fitness
        last[r] = best_k
        
    change_points = []
    curr = n
    while curr > 0:
        change_points.append(curr)
        curr = last[curr]
    change_points.reverse()
    
    clusters = []
    start = 0
    for cp in change_points:
        clusters.append(flat_data[start:cp])
        start = cp
        
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in clusters if c]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 10. Head/Tail Breaks 首尾斷裂法 (修正尾部數據丟失與未分群 Bug)
# ==============================================================================
def head_tail_breaks(flat_data):
    def recursive_break(data, all_clusters):
        if len(data) <= 1:
            if data: all_clusters.append(data)
            return
        mu = sum(data) / len(data)
        tail = [x for x in data if x <= mu]
        head = [x for x in data if x > mu]
        
        # 🎯 修正點：將尾部（Tail）完整保留至當前層級群中，避免數據被吞噬丟失
        if tail: all_clusters.append(tail)
        
        if not head or (len(head) / len(data) > 0.40):
            if head: all_clusters.append(head)
            return
        else:
            recursive_break(head, all_clusters)

    all_clusters = []
    recursive_break(sorted(flat_data), all_clusters)
    
    # 🎯 合併相同數值的微小次群以確保結構清晰
    merged_map = {}
    for clus in all_clusters:
        h_mean = round(sum(clus)/len(clus), 4)
        if h_mean not in merged_map: merged_map[h_mean] = []
        merged_map[h_mean].extend(clus)
        
    result = [{"value": k, "elements": sorted(v)} for k, v in merged_map.items()]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 11. KDE Valleys 核密度山谷硬切 (修正數據網格取樣溢出 Bug)
# ==============================================================================
def kde_valleys(flat_data):
    n = len(flat_data)
    if len(set(flat_data)) <= 1: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    std = math.sqrt(sum((x - sum(flat_data)/n)**2 for x in flat_data) / n) or 1.0
    bw = 0.9 * std * (n ** -0.2)
    if bw < 1e-2: bw = 1e-2
    
    grid_size = 300
    # 🎯 修正點：將格點網格範圍向外嚴格拓寬，防止極值數據在切分時發生索引漂移
    r_min, r_max = min(flat_data) - bw*3, max(flat_data) + bw*3
    grid = [r_min + i * (r_max - r_min) / (grid_size - 1) for i in range(grid_size)]
    
    density = []
    for g in grid:
        d = sum(math.exp(-((g - x) ** 2) / (2 * (bw ** 2))) for x in flat_data) / (n * bw * math.sqrt(2 * math.pi))
        density.append(d)
        
    valleys = []
    for i in range(1, grid_size - 1):
        if density[i] < density[i-1] and density[i] < density[i+1]:
            valleys.append(grid[i])
    valleys.sort()
    
    clusters = []
    curr_cluster = []
    # 🎯 修正點：完全依照山谷邊界進行多段式切分
    for x in sorted(flat_data):
        if valleys and x > valleys[0]:
            if curr_cluster:
                clusters.append(curr_cluster)
                curr_cluster = []
            while valleys and x > valleys[0]:
                valleys.pop(0)
        curr_cluster.append(x)
    if curr_cluster: clusters.append(curr_cluster)
        
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in clusters if c]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 12. HDBSCAN 一維自適應密度 (修正互達距離與 Lambda 生存核心)
# ==============================================================================
def optimized_hdbscan(flat_data):
    n = len(flat_data)
    if n <= 2: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    sorted_data = sorted(flat_data)
    # 🎯 修正點：實作標準一維互達距離（Mutual Reachability）架構
    core_dists = []
    for i in range(n):
        dists = [abs(sorted_data[i] - sorted_data[j]) for j in range(n) if i != j]
        core_dists.append(min(dists) if dists else 1e-5)
        
    eps_levels = sorted(list(set([d for d in core_dists if d > 0])))
    if not eps_levels: eps_levels = [1e-3]
    
    best_clusters = [[x] for x in sorted_data]
    max_stability = -1
    
    for eps in eps_levels[:20]:
        clusters = []
        curr = [sorted_data[0]]
        for i in range(n - 1):
            # 依據互達鄰域進行凝聚
            m_dist = max(sorted_data[i+1] - sorted_data[i], core_dists[i], core_dists[i+1])
            if m_dist <= eps * 2.0:
                curr.append(sorted_data[i+1])
            else:
                clusters.append(curr)
                curr = [sorted_data[i+1]]
        clusters.append(curr)
        
        # 🎯 修正點：利用 Lambda 的生存跨度壽命估算集群穩定度（Stability）
        stability = sum(len(c) * (1.0 / (eps + 1e-6)) for c in clusters if len(c) > 1)
        if stability > max_stability:
            max_stability = stability
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 13. V12 生死迭代演算法 (拓撲連續保護優化完全體)
# ==============================================================================
def v12_clustering(flat_data):
    pure_data = flatten_list(flat_data)
    if len(pure_data) == 1: return [[pure_data[0]]], 0.0
    gaps = [abs(pure_data[i+1] - pure_data[i]) for i in range(len(pure_data) - 1)]
    threshold = get_median(gaps) if gaps else 0.0
    
    clusters = []
    current = [pure_data[0]]
    for i in range(len(gaps)):
        if (gaps[i] - 1e-9) <= threshold: current.append(pure_data[i+1])
        else: clusters.append(current); current = [pure_data[i+1]]
    clusters.append(current)
    return clusters, threshold

def find_single_generation_cores(data_pool):
    current_alive = sorted(flatten_list(data_pool))
    generation_dead = []
    while True:
        if not current_alive: return [], []
        clusters, _ = v12_clustering(current_alive)
        max_size = max(len(c) for c in clusters)
        max_size_clusters = [c for c in clusters if len(c) == max_size]
        
        if len(clusters) == 2:
            next_alive = []
            for c in clusters:
                if len(c) == max_size: next_alive.extend(flatten_list(c))
                else: generation_dead.extend(flatten_list(c))
            next_alive = sorted(next_alive)
            re_clusters, _ = v12_clustering(next_alive)
            if len(re_clusters) != 1:
                return [{"value": sum(flatten_list(rc)) / len(flatten_list(rc)), "elements": flatten_list(rc)} for rc in re_clusters], sorted(generation_dead)
            elif len(re_clusters) == 1:
                return [{"value": sum(flatten_list(re_clusters)) / len(flatten_list(re_clusters)), "elements": flatten_list(re_clusters)}], sorted(generation_dead)

        if len(clusters) == 1 or len(max_size_clusters) == len(clusters):
            return [{"value": sum(flatten_list(c)) / len(flatten_list(c)), "elements": flatten_list(c)} for c in clusters], sorted(generation_dead)

        next_alive = []
        for c in clusters:
            if len(c) == max_size: next_alive.extend(flatten_list(c))
            else: generation_dead.extend(flatten_list(c))
        current_alive = sorted(next_alive)

def split_by_original_continuity(core_elements, original_data):
    if len(core_elements) <= 1: return [core_elements]
    try:
        start_idx = original_data.index(core_elements[0])
        end_idx = len(original_data) - 1 - original_data[::-1].index(core_elements[-1])
    except ValueError: return [core_elements]
        
    sub_cores = []
    current_sub = [core_elements[0]]
    core_ptr = 1 
    
    for i in range(start_idx + 1, end_idx + 1):
        if core_ptr >= len(core_elements): break
        target = original_data[i]
        if target == core_elements[core_ptr]:
            current_sub.append(target)
            core_ptr += 1
        else:
            if current_sub: sub_cores.append(current_sub)
            while core_ptr < len(core_elements) and original_data[i] > core_elements[core_ptr]: core_ptr += 1
            if core_ptr < len(core_elements):
                current_sub = [core_elements[core_ptr]]
                core_ptr += 1
            else: current_sub = []
                
    if current_sub: sub_cores.append(current_sub)
    return sub_cores

def run_v12(flat_data):
    all_cores = []
    global_dead_zone = sorted(flatten_list(flat_data))
    original_data = sorted(flatten_list(flat_data))
    
    while global_dead_zone:
        cores, dead_data = find_single_generation_cores(global_dead_zone)
        for core in cores:
            elements = core['elements']
            split_elements_list = split_by_original_continuity(elements, original_data)
            for sub_elements in split_elements_list:
                if sub_elements:
                    all_cores.append({
                        "value": sum(sub_elements) / len(sub_elements),
                        "elements": sub_elements
                    })
        global_dead_zone = sorted(flatten_list(dead_data))
    return sorted(all_cores, key=lambda x: x['value'])

# ==============================================================================
# 格式化輸出工具
# ==============================================================================
def print_method_result(title, algorithm_func, clusters, flat_data, debug_info=""):
    # 調用公平對稱的 Gap Statistic 計算機
    gap_score = calculate_gap_statistic(algorithm_func, clusters, flat_data)
    
    print(f"\n🎬 【{title}】共提煉出 {len(clusters)} 個群 {debug_info}")
    print(f"  📊 該演算法最終分群的全域『 🌟 Gap Statistic 』:  {gap_score:.4f}")
    
    for idx, c in enumerate(clusters):
        elements_str = ", ".join([str(round(x, 1)) for x in c['elements']])
        print(f"    📦 Core #{idx} -> 平均值: {c['value']:.2f} | 包含元素: [{elements_str}]")
    print("-" * 60)

def main():
    print("==================================================================")
    print("【👑 諸神黃昏：全自動優化尋參分群觀測器 V6.0 (邏輯修正公平對決版)】")
    print("==================================================================")
    while True:
        user_input = input("\n請輸入您的數列 (或輸入 'exit' 結束): ").strip()
        if user_input.lower() == 'exit' or user_input == '': break
        flat_data = parse_input_string(user_input)
        if not flat_data: continue
        print(f"\n🚀 輸入的原始數據 (計 {len(flat_data)} 個) : {flat_data}")
        print("=" * 60)
        
        # 1. K-Means
        km_res, km_k = optimized_kmeans(flat_data)
        print_method_result("1. 傳統 K-Means (分位數優化初始化)", optimized_kmeans, km_res, flat_data, f"(最佳 K={km_k})")
        
        # 2. GMM
        gmm_res, gmm_k = optimized_gmm(flat_data)
        print_method_result("2. 傳統 GMM 高斯混合 (數值穩定修正)", optimized_gmm, gmm_res, flat_data, f"(最佳 K={gmm_k})")
        
        # 3. DBSCAN
        db_res, db_eps = optimized_dbscan(flat_data)
        print_method_result("3. 傳統 DBSCAN 密度分群 (標準蔓延機制)", optimized_dbscan, db_res, flat_data, f"(半徑 eps={db_eps:.2f})")
        
        # 4. Mean Shift
        ms_res, ms_bw = run_mean_shift(flat_data)
        print_method_result("4. 傳統 Mean Shift (標準高斯核優化)", run_mean_shift, ms_res, flat_data, f"(頻寬 bw={ms_bw:.2f})")
        
        # 5. AGNES
        ag_res, ag_k = optimized_agnes(flat_data)
        print_method_result("5. 傳統 AGNES 階層分群 (孤立點懲罰修正)", optimized_agnes, ag_res, flat_data, f"(最佳 K={ag_k})")
        
        # 6. Jenks
        jk_res, jk_k = optimized_jenks(flat_data)
        print_method_result("6. 一維專用 Jenks 自然裂點法 (微擾防死鎖)", optimized_jenks, jk_res, flat_data, f"(最佳 K={jk_k})")
        
        # 7. Ckmeans
        dp_res, dp_k = ckmeans_1d_dp(flat_data)
        print_method_result("7. 🌟 一維最優 K-Means (L2 平方和指標一致)", ckmeans_1d_dp, dp_res, flat_data, f"(最佳 K={dp_k})")
        
        # 8. Bayesian Blocks
        bb_res, bb_k = bayesian_blocks(flat_data)
        print_method_result("8. 🌟 Bayesian Blocks (微幅微擾無損變點偵測)", bayesian_blocks, bb_res, flat_data, f"(區塊數 K={bb_k})")
        
        # 9. Head/Tail Breaks
        ht_res, ht_k = head_tail_breaks(flat_data)
        print_method_result("9. 🌟 Head/Tail Breaks (尾部防丟失修正)", head_tail_breaks, ht_res, flat_data, f"(遞迴切分 K={ht_k})")
        
        # 10. KDE Valleys
        kde_res, kde_k = kde_valleys(flat_data)
        print_method_result("10. 🌟 KDE Valleys (網格拓寬防漂移切分)", kde_valleys, kde_res, flat_data, f"(捕捉山谷 K={kde_k})")
        
        # 11. HDBSCAN
        hdb_res, hdb_k = optimized_hdbscan(flat_data)
        print_method_result("11. 🌟 HDBSCAN (一維互達距離生存樹簡化)", optimized_hdbscan, hdb_res, flat_data, f"(穩健核心 K={hdb_k})")
        
        # 12. V12
        v12_res = run_v12(flat_data)
        print_method_result("12. 👑 你的 V12 生死迭代演算法 (原生自適應解構)", run_v12, v12_res, flat_data, "")

if __name__ == "__main__":
    main()
     