def dp_pseudo_poly_knapsack(weights, values, max_weight):
    """
    Pseudo Polynomial Algorithm solving the 0-1 Knapsack problem for an individual player
    """
    item_list = list(weights.keys())
    n = len(item_list)

    # DP table: (n+1) x (max_weight+1)
    dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        item = item_list[i - 1]
        w_i = weights[item]
        v_i = values[item]
        for w in range(max_weight + 1):
            if w_i > w:
                dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - w_i] + v_i)

    # Recover the selected items
    selected_items = set()
    w = max_weight
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            item = item_list[i-1]
            selected_items.add(item)
            w -= weights[item]

    # Maximum total value
    max_value = dp[n][max_weight]

    return max_value, selected_items
