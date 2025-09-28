调整视觉词坐标的计算方案，使用与其关联patch在x和y轴上坐标的中位数作为视觉词的坐标。（之前的方案为计算均值）

计算中位数的代码如下：
```
    # 以中位数的方式计算每一个邻近单词的2维坐标
    frequence_map = unique_words_index.unsqueeze(-1) == all_word_index.unsqueeze(0)
    query_points_pos = query_points.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 2)
    
    x_list = []
    y_list = []
    for i in range(frequence_map.shape[0]):
        temp = query_points_pos[frequence_map[i]]
        x = torch.median(temp[:,0])
        y = torch.median(temp[:,1])
        x_list.append(x)
        y_list.append(y)
    x_list = torch.stack(x_list)
    y_list = torch.stack(y_list)
    all_words_pos_x = torch.zeros(2048).to(query_points.device).scatter_add_(dim=0, index=unique_words_index,src=x_list).unsqueeze(-1)
    all_words_pos_y = torch.zeros(2048).to(query_points.device).scatter_add_(dim=0, index=unique_words_index,src=y_list).unsqueeze(-1)
```
