import json, os

outdir = r'C:\Users\wsy\Desktop\AFAC\out\train_val\checkpoint-3000-result'
alldata = []
with open(os.path.join(outdir, 'generated_predictions.json'), 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        js = json.loads(line)
        alldata.append(js)
with open(os.path.join(outdir, '小毛吴导_result.json'), 'w+', encoding='utf-8') as f :

    for i in range(600):
        info_result, event_result = [], []
        #获取该条基金的结果
        tmp = alldata[i*10:(i+1)*10]
        #获取前5条info的结果
        for info in tmp[:5]:
            info_result.append(info['predict'])
        #获取前五条利好事件结果
        for info in tmp[5:]:
            event_result.append(info['predict'])
        
        data = {'id':str(i), "info_result":info_result, "event_result":event_result}
        jsondata = json.dumps(data, indent=0, ensure_ascii=False).replace('\n','') + '\n'
        f.write(jsondata)

print('f')

