import torch

def beibao_zeroone(weights, vals, bag):
    dp = [[0] * (bag+1) for _ in range(len(weights))]

    #初始化第一行，第一列
    for j in range(bag+1):
        dp[0][j] = vals[0] if j >= weights[0] else 0
    for i in range(len(weights)):
        dp[i][0] = 0
    
    for i in range(1, len(weights)):
        for j in range(1, 1+bag):
            #如果能放下物品i，两种可选择的状态，放或者不放
            if j >= weights[i]:
                dp[i][j] = max(dp[i-1][j-weights[i]] + vals[i], dp[i-1][j])
            #如果不能放下物品i，只有一种状态，不放
            else:
                dp[i][j] = dp[i-1][j]
    return dp[-1][-1]

weights = [1, 3, 4]
vals = [15, 20, 30]
bag = 4
result = beibao_zeroone(weights, vals, bag)
print(result)