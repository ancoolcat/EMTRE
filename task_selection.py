import pulp

def HSP(bian, N=6, k=3):
    E = [(i, j) for i in range(N) for j in range(i+1,N)]

    W = {(i, j): bian[i][j] for (i, j) in E}

    problem = pulp.LpProblem("The_Heaviest_K_Subgraph", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", range(N), cat="Binary")
    y = pulp.LpVariable.dicts("y", E, cat="Binary")


    problem += pulp.lpSum(W[(i, j)] * y[(i, j)] for (i, j) in E)

    problem += pulp.lpSum(x[i] for i in range(N)) == k

    for (i, j) in E:
        problem += y[(i, j)] <= x[i]

        problem += y[(i, j)] <= x[j]

        problem += y[(i, j)] >= 0

    for i in range(N):
        temp=0
        for (ii, j) in E:
            if ii==i or j==i:
                temp+=y[(ii, j)]
        problem += temp==((k-1)*x[i])

    problem.solve()

    selected_vertices = [i for i in range(N) if x[i].value() == 1]

    print("Selected vertices:", selected_vertices)

    for i in selected_vertices:
        li=[]
        for j in selected_vertices:
            li.append(bian[i][j])
        print(li)
    return selected_vertices


# bian = [
#         [0, 1, 2, 3, 1, 1],
#         [1, 0, 1, 2, 1, 1],
#         [2, 1, 0, 44, 1, 5],
#         [3, 2, 44, 0, 11, 22],
#         [1, 1, 1, 11, 0, 6],
#         [1, 1, 5, 22, 6, 0]]
# # bian=[[0, 2, 3, 4],
# #       [2, 0, 1, 5],
# #       [3, 1, 0, 6],
# #       [4, 5, 6, 0]]#4
# selected_vertices=HSP(bian,N=6,k=3)
# print("result:"+str(selected_vertices))
