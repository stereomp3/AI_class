import queue


m = [['0', '0', '0', '0', '0', '0', '0', '0'],
     ['0', '0', '.', '0', '.', '0', '0', '0'],
     ['M', '.', '.', '.', '.', '0', '0', '0'],
     ['0', '.', '0', '0', '0', '0', '0', '0'],
     ['0', '.', '.', '.', '.', '.', '0', '0'],
     ['0', '0', '0', '0', '0', '.', '0', '0'], ]

Mx = 0
My = 0
col = len(m[0])
row = len(m)
for x in range(row):
    for y in range(col):
        if m[x][y] == "M":
            Mx = x
            My = y
print(Mx)
print(My)

q = queue.Queue()
q.put((Mx, My, 0))  # x,y,step

# build m
while not q.empty():
    pos = q.get()  # pos[0] = x, pos[1] = y
    print(pos)
    if pos[0] + 1 < row:  # down
        if m[pos[0] + 1][pos[1]] == '.':  # is road
            q.put((pos[0] + 1, pos[1], pos[2] + 1))
            m[pos[0] + 1][pos[1]] = pos[2] + 1

    if pos[0] - 1 >= 0:  # up
        if m[pos[0] - 1][pos[1]] == '.':
            q.put((pos[0] - 1, pos[1], pos[2] + 1))
            m[pos[0] - 1][pos[1]] = pos[2] + 1
    if pos[1] + 1 < col:  # right
        if m[pos[0]][pos[1] + 1] == '.':
            q.put((pos[0], pos[1] + 1, pos[2] + 1))
            m[pos[0]][pos[1] + 1] = pos[2] + 1
    if pos[1] - 1 >= 0:  # left
        if m[pos[0]][pos[1] - 1] == '.':
            q.put((pos[0], pos[1] - 1, pos[2] + 1))
            m[pos[0]][pos[1] - 1] = pos[2] + 1

for r in m:
    print(r)
shortest = -1
for x in range(row):
    for y in range(col):
        if x == 0 or x == row-1 or y == 0 or y == col-1:  # check board
            if m[x][y] != "0" and m[x][y] != "." and m[x][y] != "M":
                if shortest > int(m[x][y]) or shortest == -1:
                    shortest = int(m[x][y])
if shortest == -1:
    print("No Way to go out.")
else:
    print(shortest)
