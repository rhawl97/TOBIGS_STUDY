def Q1(triangle):
    dp = []
    for row in triangle:
        lst = []
        if len(dp) == 0: #초기값 지정 첫 행 숫자!
            lst.append(row[0])
        else:
            for idx, num in enumerate(row):
                if idx == 0: #가장 왼쪽에 있는 숫자일 때
                    lst.append(dp[-1][idx] + num) #이전 행 맨 처음 숫자 바로 더하기
                elif idx == len(row)-1:  #가장 오른쪽에 있는 숫자일 때
                    lst.append(dp[-1][-1] + num)  #이전 행 맨 마지막 숫자 바로 더하기
                else:  #양 끝이 아닌 중간에 위치한 숫자일 경우
                    lst.append(max(dp[-1][idx-1],dp[-1][idx]) + num) #이전 행 같은 위치 두 숫자 중 최대값으로 더하기
        dp.append(lst) #행에서 최대값으로 계산된 합 저장
    return max(dp[-1])  #최종 값 반환

a = [[7],[3,8],[8,1,0],[2,7,4,4],[4,5,2,6,5]]
print(Q1(a))
