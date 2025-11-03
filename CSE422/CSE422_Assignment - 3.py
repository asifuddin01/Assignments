
# Task 1
filename = "/content/drive/MyDrive/422_Assignment 02/Assignement03_InputFile_1- .txt"


with open(filename, "r") as file:
    lines = file.readlines()


pool_line = lines[0].strip()
target_line = lines[1].strip()
sid_line = lines[2].strip()


pool = pool_line.split(",")
target = list(target_line)
sId = list(map(int, sid_line.split()))


# Extraction of Weights
target_len = len(target)
weight = sId[-target_len:]



def calculate_utility(gene_seq, target_seq, wght_list):
    score = 0
    max_len = max(len(gene_seq), len(target_seq))

    for i in range(max_len):

        if i < len(gene_seq):
            g_val = ord(gene_seq[i])

        else:
            g_val = 0


        if i < len(target_seq):
            t_val = ord(target_seq[i])

        else:
            t_val = 0


        if i < len(wght_list):
            w = wght_list[i]

        else:
            w = 1

        diff = abs(g_val - t_val)
        score += w * diff

    return -score  # Negative Agent 1 maximizes similarity


def minimax(pool, gene_seq, maximizing, alpha, beta):

    if not pool:
        return calculate_utility(gene_seq, target, weight), gene_seq

    best_sequence = None

    if maximizing:
        max_eval = float('-inf')

        for i in range(len(pool)):
            new_pool = pool[:i] + pool[i+1:]
            new_gene = gene_seq + [pool[i]]
            eval_score, eval_seq = minimax(new_pool, new_gene, False, alpha, beta)

            if eval_score > max_eval:
                max_eval = eval_score
                best_sequence = eval_seq

            alpha = max(alpha, eval_score)

            if beta <= alpha:
                break

        return max_eval, best_sequence

    else:

        min_eval = float('inf')

        for i in range(len(pool)):
            new_pool = pool[:i] + pool[i+1:]
            new_gene = gene_seq + [pool[i]]
            eval_score, eval_seq = minimax(new_pool, new_gene, True, alpha, beta)

            if eval_score < min_eval:
                min_eval = eval_score
                best_sequence = eval_seq

            beta = min(beta, eval_score)

            if beta <= alpha:
                break

        return min_eval, best_sequence

# Game start
final_score, final_sequence = minimax(pool, [], True, float('-inf'), float('inf'))


print("Best gene sequence generated:", ''.join(final_sequence))
print("Utility score:", final_score)









#task 2
filename = "/content/drive/MyDrive/422_Assignment 02/Assignement03_InputFile_1- .txt"


with open(filename, "r") as file:
    lines = file.readlines()


pool_line = lines[0].strip()
target_line = lines[1].strip()
sid_line = lines[2].strip()

pool = [n.strip() for n in pool_line.split(",")]
target = list(target_line)
sId = list(map(int, sid_line.split()))

# Extraction of Weights
weights = sId[-len(target):]
booster_mul = (sId[0] * 10 + sId[1]) / 100



def utility(gene, target, weights, booster_index=None, booster=None):
    total = 0
    max_len = max(len(gene), len(target))

    for i in range(max_len):

        if i < len(gene):
            g_val = ord(gene[i])

        else:
            g_val = 0


        if i < len(target):
            t_val = ord(target[i])

        else:
            t_val = 0


        if i < len(weights):
            w = weights[i]

        else:
            w = 1

        # Apply booster from booster index onward
        if booster_index is not None and i >= booster_index:
            w *= booster

        diff = abs(g_val - t_val)
        total += w * diff

    return -total


# Minimax with alpha-beta pruning
def minimax(pool, gene, maximizing, alpha, beta, booster_index=None):
    if not pool:
        return utility(gene, target, weights, booster_index, booster_mul), gene

    best_gene = None

    if maximizing:
        max_score = float('-inf')

        for i in range(len(pool)):
            picked = pool[i]
            next_pool = pool[:i] + pool[i+1:]
            new_gene = gene + [picked]

            booster_used = booster_index
            if picked == 'S' and booster_index is None:
                booster_used = len(gene)  # Booster starts from this index

            score, result_gene = minimax(next_pool, new_gene, False, alpha, beta, booster_used)

            if score > max_score:
                max_score = score
                best_gene = result_gene

            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return max_score, best_gene

    else:
        min_score = float('inf')

        for i in range(len(pool)):
            picked = pool[i]
            next_pool = pool[:i] + pool[i+1:]
            new_gene = gene + [picked]

            score, result_gene = minimax(next_pool, new_gene, True, alpha, beta, booster_index)

            if score < min_score:
                min_score = score
                best_gene = result_gene

            beta = min(beta, score)
            if beta <= alpha:
                break

        return min_score, best_gene


# Ensuring 'S' exists in pool
if 'S' not in pool:
    pool.append('S')


# Making a new list without 'S' in it
normal_pool = []

for n in pool:
    if n != 'S':
        normal_pool.append(n)


normal_gene = []
normal_alpha = float('-inf')
normal_beta = float('inf')
normal_score, best_normal_gene = minimax(normal_pool, normal_gene, True, normal_alpha, normal_beta)



# Making a copy of the original pool (with 'S' in it)
special_pool = []

for n in pool:
    special_pool.append(n)


special_gene = []
special_alpha = float('-inf')
special_beta = float('inf')
special_score, best_special_gene = minimax(special_pool, special_gene, True, special_alpha, special_beta)

# result
if special_score > normal_score:
    print("YES")
else:
    print("NO")

print("With special nucleotide")
print("Best gene sequence generated:", ''.join(special_gene) + ",")
print("Utility score:", round(special_score, 2))