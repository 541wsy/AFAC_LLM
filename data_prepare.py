import json

with open(r'C:\Users\wsy\Desktop\AFAC\rawdata\训练集.json', 'rb') as f:
    train_text = json.load(f)

hangye = train_text['行业']
weidu = train_text['维度']
zhi = train_text['值']
huashu = train_text['话术']

#检查训练集的不同keys id是否相同
hangye_ids = list(map(lambda x: int(x), list(hangye.keys())))
weidu_ids = list(map(lambda x: int(x), list(weidu.keys())))
zhi_ids = list(map(lambda x: int(x), list(zhi.keys())))
huashu_ids = list(map(lambda x: int(x), list(huashu.keys())))

hangye_ids.sort(); weidu_ids.sort(); zhi_ids.sort(); huashu_ids.sort()

assert len(hangye_ids) == len(weidu_ids) == len(zhi_ids) == len(huashu_ids), 'The keys of trainset have diffent lengths'
for i in range(len(hangye_ids)):
    if hangye_ids[i] != weidu_ids[i] != zhi_ids[i] != huashu_ids[i]:
        print('Error')
        break

all_data = []
# with open('train.json', 'w+', encoding='utf-8') as f:
#     for id in list(map(lambda x: str(x), hangye_ids)):
#         hang_text = hangye[id]
#         wei_text = weidu[id]
#         zhi_text = zhi[id]
#         hua_text = huashu[id]

#         if isinstance(hang_text, list):
#             if len(hang_text) == 1:
#                 hang_text = hang_text[0]
#             else:
#                 hang_text = sum(hang_text, [])[0]
#         if isinstance(wei_text, list):
#             if len(wei_text) == 1:
#                 wei_text = wei_text[0]
#             else:
#                 wei_text = sum(wei_text, [])[0]
#         if isinstance(zhi_text, list):
#             if len(zhi_text) == 1:
#                 zhi_text = zhi_text[0]
#             else:
#                 zhi_text = sum(zhi_text, [])[0]
#         if isinstance(hua_text, list):
#             if len(hua_text) == 1:
#                 hua_text = hua_text[0]
#             else:
#                 hua_text = sum(hua_text, [])[0]

#         text = '#'.join(['行业*'+hang_text, "维度*"+wei_text, "值*"+zhi_text])
#         answer = hua_text
#         rowdata = {'text': text, 'answer':answer}
#         json_rowdata = json.dumps(rowdata, indent=0, ensure_ascii=False).replace('\n','') + '\n'
#         f.write(json_rowdata)
print('f')

