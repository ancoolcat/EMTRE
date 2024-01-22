import random
import heapq


def a_res_based_sampling(weighted_features, N_sam):
    result_tasks = []

    for i in range(N_sam):
        m =random.randint(1, len(weighted_features) - 1)
        I_pre = []

        for a, weight in enumerate(weighted_features):
            k_a = random.random() ** (1 / weight)

            if len(I_pre) < m:
                heapq.heappush(I_pre, (k_a, a))
            else:
                min_k_a_prime, min_k_a_index = heapq.heappop(I_pre)
                if k_a > min_k_a_prime:
                    heapq.heappush(I_pre, (k_a, a))
                else:
                    heapq.heappush(I_pre,(min_k_a_prime,min_k_a_index))

        result_tasks.append([index for _, index in I_pre])
        print("task_" + str(i) + " is completed")

    return result_tasks



# weighted_features = [0.5, 1.0, 1.5, 2.0]
# N_sam = 3
# output = a_res_based_sampling_optimized(weighted_features, N_sam)
# print(output)

