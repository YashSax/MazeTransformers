sz = 16

arr = [[0 for _ in range(sz)] for __ in range(sz)]
arr[0] = [1 for _ in range(sz)]
for i in range(sz):
    arr[i][0] = 1

for i in range(1, sz):
    for j in range(1, sz):
        arr[i][j] = arr[i - 1][j] + arr[i][j - 1]

print(arr[-1][-1])