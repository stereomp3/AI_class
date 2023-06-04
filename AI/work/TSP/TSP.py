import math
import random
input = {1: [0, 0],
         2: [1, 1],
         3: [2, 2],
         4: [3, 3],
         5: [1, 0],
         6: [0, 1],
         7: [1, 2],
         8: [2, 1],
         9: [1, 3],
         10: [3, 1],
         11: [2, 3],
         12: [3, 2], }


class Graph:
    def __init__(self, graph):
        self.g = graph

    def getGraph(self):
        return self.g


class TSP:
    def __init__(self, graph):
        self.graph = graph
        self.score = 0
        self.order = []
    def calculate(self, tolerate,times):
        self.ListInit()
        l = len(self.order)
        pre_score = self.score
        pre_order = self.order
        tol = tolerate
        while True:
            self.score = 0
            # self.order = self.swap()
            for i in range(l - 1):
                now_pos = self.graph[self.order[i]]
                next_pos = self.graph[self.order[i + 1]]
                h = self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
                for _ in range(times):
                    new_order = self.swap(i+1, random.randint(i+1, l-1))
                    # print(new_order)
                    now_pos = self.graph[new_order[i]]
                    next_pos = self.graph[new_order[i + 1]]
                    nh = self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
                    # print("nh", nh)
                    # print("h", h)
                    if h > nh:
                        self.order = new_order
                        # print(new_order)
                self.score += self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
            print(self.score)
            if not tol:
                return [pre_score, pre_order]

            if self.score < pre_score:
                if not tol:
                    return [self.score, self.order]
                else:
                    print(tol)
                    pre_order = self.order
                    pre_score = self.score
                    tol = tolerate
            elif not pre_score:
                pre_score = self.score
            else:
                tol -= 1


    def distence(self, x, y):
        return math.sqrt(x ** 2 + y ** 2)

    def swap(self, a, b):
        l = len(self.order)
        new_order = []
        for i in range(l):
            if a == i:
                new_order.append(self.order[b])
            elif b == i:
                new_order.append(self.order[a])
            else:
                new_order.append(self.order[i])

        return new_order

    def ListInit(self):
        for k, v in self.graph.items():
            self.order.append(k)


graph = Graph(input)
ans = TSP(graph.getGraph())
print(graph.getGraph())
print(ans.calculate(10, 100))
# print(ans.order)
# print(ans.swap(1,2))
# print(ans.swap(2,2))
