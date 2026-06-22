```python
import re
import math
import random
from collections import defaultdict

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

# ==============================================================================
# 2. 傳統 K-Means
# ==============================================================================
def run_kmeans_core(flat_data, k):
    n = len(flat_data)
    if k >= n: return [[x] for x in flat_data], flat_data[:]
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
# 3. 傳統 GMM 高斯混合
# ==============================================================================
def pdf(x, mu, var):
    if var < 1e-6: var = 1e-6
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
        for i in range(n):
            total_density = 0.0
            for j in range(k):
                resp[i][j] = weights[j] * pdf(flat_data[i], means[j], vars_[j])
                total_density += resp[i][j]
            if total_density < 1e-12:
                resp[i] = [1.0 / k] * k
            else:
                for j in range(k): resp[i][j] /= total_density
        for j in range(k):
            N_j = sum(resp[i][j] for i in range(n))
            if N_j < 1e-4: N_j = 1e-4
            means[j] = sum(resp[i][j] * flat_data[i] for i in range(n)) / N_j
            vars_[j] = sum(resp[i][j] * (flat_data[i] - means[j])**2 for i in range(n)) / N_j
            if vars_[j] < 1e-4: vars_[j] = 1e-4
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
# 4. 傳統 DBSCAN
# ==============================================================================
def optimized_dbscan(flat_data):
    if not flat_data: return [], 0.0
    n = len(flat_data)
    if n < 2: return [{"value": flat_data[0], "elements": flat_data}], 0.0
    
    dists = [abs(flat_data[i+1] - flat_data[i]) for i in range(n - 1)]
    eps = sum(dists) / len(dists) * 0.5
    if eps == 0: eps = 1e-4
    
    labels = [-1] * n
    cluster_id = 0
    
    for i in range(n):
        if labels[i] != -1: continue
        neighbors = [j for j in range(n) if abs(flat_data[i] - flat_data[j]) <= eps]
        
        if len(neighbors) < 2:
            continue
            
        labels[i] = cluster_id
        queue = [idx for idx in neighbors if idx != i]
        
        while queue:
            current_idx = queue.pop(0)
            if labels[current_idx] == -1:
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
            for e in elements: result.append({"value": e, "elements": [e]})
        else:
            result.append({"value": sum(elements) / len(elements), "elements": sorted(elements)})
    return sorted(result, key=lambda x: x['value']), eps

# ==============================================================================
# 5. 傳統 Mean Shift
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
        closest_peak = min(unique_peaks, key=lambda p: abs(p - peak))
        if closest_peak not in clusters_dict: clusters_dict[closest_peak] = []
        clusters_dict[closest_peak].append(pt)
        
    result = [{"value": sum(clus) / len(clus), "elements": sorted(clus)} for clus in clusters_dict.values() if clus]
    return sorted(result, key=lambda x: x['value']), bandwidth

# ==============================================================================
# 6. 傳統 AGNES 階層分群
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
# 7. 一維專用 Jenks 自然裂點法
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
# 8. 一維最優 K-Means
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
    best_score = float('inf')
    
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
            
        total_ss = dp[k][n]
        if total_ss < best_score:
            best_score = total_ss
            best_k = k
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), best_k

# ==============================================================================
# 9. Bayesian Blocks 貝氏區塊法
# ==============================================================================
def bayesian_blocks(flat_data):
    n = len(flat_data)
    if n <= 1: return [{"value": flat_data[0], "elements": flat_data}], 1
    
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
# 10. Head/Tail Breaks 首尾斷裂法
# ==============================================================================
def head_tail_breaks(flat_data):
    def recursive_break(data, all_clusters):
        if len(data) <= 1:
            if data: all_clusters.append(data)
            return
        mu = sum(data) / len(data)
        tail = [x for x in data if x <= mu]
        head = [x for x in data if x > mu]
        
        if tail: all_clusters.append(tail)
        
        if not head or (len(head) / len(data) > 0.40):
            if head: all_clusters.append(head)
            return
        else:
            recursive_break(head, all_clusters)

    all_clusters = []
    recursive_break(sorted(flat_data), all_clusters)
    
    merged_map = {}
    for clus in all_clusters:
        h_mean = round(sum(clus)/len(clus), 4)
        if h_mean not in merged_map: merged_map[h_mean] = []
        merged_map[h_mean].extend(clus)
        
    result = [{"value": k, "elements": sorted(v)} for k, v in merged_map.items()]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 11. KDE Valleys 核密度山谷硬切
# ==============================================================================
def kde_valleys(flat_data):
    n = len(flat_data)
    if len(set(flat_data)) <= 1: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    std = math.sqrt(sum((x - sum(flat_data)/n)**2 for x in flat_data) / n) or 1.0
    bw = 0.9 * std * (n ** -0.2)
    if bw < 1e-2: bw = 1e-2
    
    grid_size = 300
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
# 12. HDBSCAN 一維自適應密度
# ==============================================================================
def optimized_hdbscan(flat_data):
    n = len(flat_data)
    if n <= 2: return [{"value": flat_data[0], "elements": flat_data}], 1
    
    sorted_data = sorted(flat_data)
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
            m_dist = max(sorted_data[i+1] - sorted_data[i], core_dists[i], core_dists[i+1])
            if m_dist <= eps * 2.0:
                curr.append(sorted_data[i+1])
            else:
                clusters.append(curr)
                curr = [sorted_data[i+1]]
        clusters.append(curr)
        
        stability = sum(len(c) * (1.0 / (eps + 1e-6)) for c in clusters if len(c) > 1)
        if stability > max_stability:
            max_stability = stability
            best_clusters = clusters
            
    result = [{"value": sum(c) / len(c), "elements": sorted(c)} for c in best_clusters if c]
    return sorted(result, key=lambda x: x['value']), len(result)

# ==============================================================================
# 13. 👑 完全體 V12 生死迭代演算法
# ==============================================================================
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

def v12_core_split_visual(pure_data, generation_id, global_initial_threshold=None, threshold_multiplier=1.5):
    if len(pure_data) <= 1: return [pure_data], 0.0
    gaps = [abs(pure_data[i+1] - pure_data[i]) for i in range(len(pure_data) - 1)]
    threshold = get_median(gaps)
    
    if len(pure_data) == 2 and global_initial_threshold is not None:
        max_allowed_threshold = global_initial_threshold * threshold_multiplier
        if threshold > max_allowed_threshold: threshold = max_allowed_threshold - 1e-9
            
    clusters = []
    current = [pure_data[0]]
    for i in range(len(gaps)):
        if (gaps[i] - 1e-9) <= threshold: current.append(pure_data[i+1])
        else:
            clusters.append(current)
            current = [pure_data[i+1]]
    clusters.append(current)
    return clusters, threshold

def find_single_generation_cores_debug(data_pool, stage_counter, global_initial_threshold=None, threshold_multiplier=1.5):
    current_alive = sorted(data_pool)
    generation_dead = []
    immediate_cores = []  
    generation_id = 1
    first_gen_threshold = None
    
    while True:
        if not current_alive: return [], [], [], None
        clusters, current_threshold = v12_core_split_visual(current_alive, generation_id, global_initial_threshold, threshold_multiplier)
        
        if stage_counter == 1 and generation_id == 1: first_gen_threshold = current_threshold
        max_size = max(len(c) for c in clusters)
        max_size_clusters = [c for c in clusters if len(c) == max_size]
        
        if len(clusters) == 2 and len(max_size_clusters) == 2:
            next_alive = []
            for c in clusters:
                if len(c) == max_size: next_alive.extend(c)
                else: generation_dead.extend(c)
            next_alive = sorted(next_alive)
            re_clusters, _ = v12_core_split_visual(next_alive, generation_id + 1, global_initial_threshold, threshold_multiplier)
            return (re_clusters if len(re_clusters) != 1 else [next_alive]), sorted(generation_dead), immediate_cores, (global_initial_threshold or first_gen_threshold)

        if len(clusters) == 1 or len(max_size_clusters) == len(clusters):
            return clusters, sorted(generation_dead), immediate_cores, (global_initial_threshold or first_gen_threshold)

        next_alive = []
        active_global_thresh = global_initial_threshold if global_initial_threshold is not None else first_gen_threshold
        limit = active_global_thresh * threshold_multiplier if active_global_thresh is not None else float('inf')

        for c in clusters:
            if len(c) == max_size: next_alive.extend(c)
            else:
                if generation_id > 1:
                    sub_c = []
                    curr_sub = [c[0]]
                    for idx in range(len(c) - 1):
                        gap = abs(c[idx+1] - c[idx])
                        if gap <= limit: curr_sub.append(c[idx+1])
                        else:
                            sub_c.append(curr_sub)
                            curr_sub = [c[idx+1]]
                    sub_c.append(curr_sub)
                    for sc in sub_c: immediate_cores.append(sc)
                else: generation_dead.extend(c)
        current_alive = sorted(next_alive)
        generation_id += 1

def run_v12_core_logic(flat_data, threshold_multiplier=1.5):
    if not flat_data: return []
    original_data = sorted(flat_data)
    global_dead_zone = sorted(flat_data)
    all_cores = []
    stage_counter = 1
    global_initial_threshold = None
    
    while global_dead_zone:
        cores, dead_data, immediate_cores, first_thresh = find_single_generation_cores_debug(global_dead_zone, stage_counter, global_initial_threshold, threshold_multiplier)
        if stage_counter == 1 and first_thresh is not None: global_initial_threshold = first_thresh
        
        for core_list in (cores + immediate_cores):
            for sub_elements in split_by_original_continuity(core_list, original_data):
                if sub_elements: all_cores.append({"value": sum(sub_elements) / len(sub_elements), "elements": sub_elements})
        if len(global_dead_zone) == len(dead_data): break
        global_dead_zone = sorted(dead_data)
        stage_counter += 1
        
    num_elements = len(original_data)
    if num_elements >= 3:
        used = [False] * num_elements
        v12_core_indices = []
        for c in all_cores:
            current_core_idx = []
            for elem in c["elements"]:
                for i in range(num_elements):
                    if not used[i] and original_data[i] == elem:
                        used[i] = True; current_core_idx.append(i); break
            v12_core_indices.append(sorted(current_core_idx))
            
        gaps = [abs(original_data[i+1] - original_data[i]) for i in range(num_elements - 1)]
        ap_blocks = []
        i = 0
        while i < len(gaps):
            j = i
            while j < len(gaps) and abs(gaps[j] - gaps[i]) < 1e-9: j += 1
            if (j - i) >= 2: ap_blocks.append(list(range(i, j + 1))); i = j
            else: i += 1
        for k in range(len(ap_blocks) - 1):
            if ap_blocks[k][-1] == ap_blocks[k+1][0]: ap_blocks[k].pop()
        belongs_to_ap = [None] * num_elements
        for block_id, block in enumerate(ap_blocks):
            for idx in block: belongs_to_ap[idx] = block_id
                
        parent = list(range(num_elements))
        def find_root(idx):
            if parent[idx] == idx: return idx
            parent[idx] = find_root(parent[idx]); return parent[idx]
        def union_nodes(idx1, idx2):
            r1, r2 = find_root(idx1), find_root(idx2)
            if r1 != r2: parent[r1] = r2

        for block in ap_blocks:
            for idx in range(len(block) - 1): union_nodes(block[idx], block[idx+1])
        for current_core_idx in v12_core_indices:
            for idx in range(len(current_core_idx) - 1):
                i1, i2 = current_core_idx[idx], current_core_idx[idx+1]
                if not (belongs_to_ap[i1] is not None and belongs_to_ap[i2] is not None and belongs_to_ap[i1] != belongs_to_ap[i2]): union_nodes(i1, i2)
                    
        components = defaultdict(list)
        for idx in range(num_elements): components[find_root(idx)].append(idx)
        all_cores = [{"value": sum(original_data[idx] for idx in sorted(ids)) / len(ids), "elements": [original_data[idx] for idx in sorted(ids)]} for ids in components.values()]
        
    return sorted(all_cores, key=lambda x: x['value'])

def run_v12(flat_data):
    cores = run_v12_core_logic(flat_data, threshold_multiplier=1.5)
    return cores, len(cores)

# ==============================================================================
# 格式化輸出工具
# ==============================================================================
def print_method_result(title, clusters, debug_info=""):
    print(f"\n🎬 【{title}】共提煉出 {len(clusters)} 個群 {debug_info}")
    for idx, c in enumerate(clusters):
        elements_str = ", ".join([str(round(x, 1)) for x in c['elements']])
        print(f"    📦 Core #{idx} -> 平均值: {c['value']:.2f} | 包含元素: [{elements_str}]")
    print("-" * 60)

def main():
    print("==================================================================")
    print("【👑 諸神黃昏：全自動優化尋參分群觀測器 V6.1 精簡極速版】")
    print("==================================================================")
    while True:
        user_input = input("\n請輸入您的數列 (或輸入 'exit' 結束): ").strip()
        if user_input.lower() == 'exit' or user_input == '': break
        flat_data = parse_input_string(user_input)
        if not flat_data: continue
        print(f"\n🚀 輸入的原始數據 (計 {len(flat_data)} 個) : {flat_data}")
        print("=" * 60)
        
        km_res, km_k = optimized_kmeans(flat_data)
        print_method_result("1. 傳統 K-Means (分位數優化初始化)", km_res, f"(最佳 K={km_k})")
        
        gmm_res, gmm_k = optimized_gmm(flat_data)
        print_method_result("2. 傳統 GMM 高斯混合 (數值穩定修正)", gmm_res, f"(最佳 K={gmm_k})")
        
        db_res, db_eps = optimized_dbscan(flat_data)
        print_method_result("3. 傳統 DBSCAN 密度分群 (標準蔓延機制)", db_res, f"(半徑 eps={db_eps:.2f})")
        
        ms_res, ms_bw = run_mean_shift(flat_data)
        print_method_result("4. 傳統 Mean Shift (標準高斯核優化)", ms_res, f"(頻寬 bw={ms_bw:.2f})")
        
        ag_res, ag_k = optimized_agnes(flat_data)
        print_method_result("5. 傳統 AGNES 階層分群 (孤立點懲罰修正)", ag_res, f"(最佳 K={ag_k})")
        
        jk_res, jk_k = optimized_jenks(flat_data)
        print_method_result("6. 一維專用 Jenks 自然裂點法 (微擾防死鎖)", jk_res, f"(最佳 K={jk_k})")
        
        dp_res, dp_k = ckmeans_1d_dp(flat_data)
        print_method_result("7. 🌟 一維最優 K-Means (L2 平方和指標一致)", dp_res, f"(最佳 K={dp_k})")
        
        bb_res, bb_k = bayesian_blocks(flat_data)
        print_method_result("8. 🌟 Bayesian Blocks (微幅微擾無損變點偵測)", bb_res, f"(區塊數 K={bb_k})")
        
        ht_res, ht_k = head_tail_breaks(flat_data)
        print_method_result("9. 🌟 Head/Tail Breaks (尾部防丟失修正)", ht_res, f"(遞迴切分 K={ht_k})")
        
        kde_res, kde_k = kde_valleys(flat_data)
        print_method_result("10. 🌟 KDE Valleys (網格拓寬防漂移切分)", kde_res, f"(捕捉山谷 K={kde_k})")
        
        hdb_res, hdb_k = optimized_hdbscan(flat_data)
        print_method_result("11. 🌟 HDBSCAN (一維互達距離生存樹簡化)", hdb_res, f"(穩健核心 K={hdb_k})")
        
        v12_res, v12_k = run_v12(flat_data)
        print_method_result("12. 👑 完全體 V12 生死迭代演算法 (AP保護與DSU熔接)", v12_res, f"(終極收斂 K={v12_k})")

if __name__ == "__main__":
    main()
