[code](climbToLine.py)

最後結果: 

* `p = [1.9000000000000015, 1.0600000000000007]`
* `loss = 0.04399992752080731 `
* `y_predicted= [1.9000000000000015, 2.960000000000002, 4.020000000000003, 5.080000000000004, 6.140000000000004]`
* `y_origin = [1.9, 3.1, 3.9, 5.0, 6.2]`

> climbToLine.py: 使用h = 0.01做爬山，使用老師原本設定的MSE做loss function，下面是主要添加的爬山程式碼

```python
def ArrayAdd(l1, l2, sub=False):  # 列表相加函數
    l1c = l1.copy()
    for i in range(len(l1c)):
        if not sub:
            l1c[i] += l2[i]
        else:
            l1c[i] -= l2[i]
    return l1c


def hillClimbing(f, hx, h=0.01):
    h_x = hx.copy()  # 深拷貝一個新的p
    while True:
        In = False  # 檢測是否到山頂
        for i in range(len(h_x)):  
            dh = []
            # 根據輸入的列表產生新的dh，這邊產生四個: [1,0]、[-1,0]、[0,1]、[0,-1]
            for num in range(len(h_x)):  
                if num == i:
                    dh.append(h)
                else:
                    dh.append(0)
            nh_x = ArrayAdd(h_x, dh)
            nh_x2 = ArrayAdd(h_x, dh, True)
            if f(nh_x) < f(h_x):  # 比較loss function，loss越小越好
                h_x = nh_x
                In = True
            if f(nh_x2) < f(h_x):
                h_x = nh_x2
                In = True
        if not In:
            break
    return h_x
```

