def Q2(m, n, gold):

    getgold = [[0 for i in range(n)]  #영행렬(이중 리스트) 생성 #획득하는 금 넣을 곳
                        for j in range(m)]

    for col in range(n-1, -1, -1): #room 열 전체(역순으로)
        for row in range(m-1, -1, -1): #room 행 전체

            #(r,c)->(r,c+1) 오른쪽 칸으로 이동할 때
            if (col == n-1):  #가장 오른쪽 칸은 더 오른쪽이 없음
                right = 0
            else:
                right = getgold[row][col+1] #오른쪽 칸 금 획득

            #(r,c)->(r+1,c) 아래 칸으로 이동할 때
            if (row == m-1):
                down = 0
            else:
                down = getgold[row+1][col]  #아래 칸 금 획득

            #(r,c)->(r+1,c+1) 오른쪽 아래 칸으로 이동할 때
            if (row == m-1 or col == n-1):
                right_down = 0
            else:
                right_down = getgold[row+1][col+1]  #오른쪽 아래 칸 금 획득

            getgold[row][col] = gold[row][col] + max(right, down, right_down) #가장 최대 개수를 가지는 방향으로

    final = getgold[0][0] #getgold 행렬에서 가장 최대값 구함
    print(getgold)
    return final

g = [[2, 1, 0],
    [2, 5, 4],
    [3, 9, 2]]

a = 3
b = 3
print(Q2(a,b,g))
