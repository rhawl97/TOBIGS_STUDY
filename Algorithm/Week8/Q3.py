# def Q3(k,size):
#     dp = [[0]*k for i in range(k)]
#
#     if k <= 2:
#         dp[i][j] = sum(size)
#     else:
#         for z in len(size-2):
#             dp[i][j] = min(dp[i][z]+dp[z+1][j]+sum(size[i~j]))
#     return(min(dp))


# import sys
# d = [10, 30, 5, 60]
# M = [[0 for x in range(4)] for y in range(4)]
# ​
# for diag in range(1, 4):
#     for i in range(1, 4-diag):
#         j = i+diag
#         M[i][j] = sys.maxsize
#         for k in range(i,j):
#             M[i][j] = min(M[i][j],
#                           M[i][k] + M[k+1][j] + d[i-1]*d[k]*d[j])
#
# print(M[1][3])

# def Q3(k,size):
#     dp = [[0]*k for i in range(k)]
#
#     for i in range(k):
#         for j in range(k-i):
#             z = i+j
#             dp[i][i+1] = size[i]+size[i+1]
#             # dp[j][j+1] = size[j] + size[j+1]
#             for s in range(i,j):
#                 dp[i][j] = min(dp[j][z], dp[j][s] + dp[s+1][i] + size[j-1]+size[s]+size[z])
#     return(dp)


# def Q3(n, sizes):
#   dp = [[0]*n for i in range(n)]
#   i = 0
#   j = n - 1
#
#   if i==j:
#       return 0
#   # if dp[i][j]!=0:
#   #     return dp[i][j]
#
#   result = min(Q3(len(sizes[:k]),sizes[:k])+Q3(len(sizes[k:]),sizes[k:]) for k in range(i,j))
#   result += sizes[j+1]-sizes[i]
#
#   dp[i][j] = result
#   return result
#
# a = 4
# size = [40,30,30,50]
# print(Q3(a,size))

m = [[1,2],[4,5]]
print(m[0][1]==2)
