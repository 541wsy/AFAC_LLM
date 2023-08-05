from transformers import AutoModel, AutoTokenizer
import json, re
from tqdm import trange
from knowledge.myknow import DocumentService

device = 'cuda:1'
model = AutoModel.from_pretrained('/data/wusiyao/chatglm/chatglm/chatglm2-6b', trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained('/data/wusiyao/chatglm/chatglm/chatglm2-6b', trust_remote_code=True)
model = model.eval()

#本地知识库
vs_path = '/data/wusiyao/chatglm/chatglm/sourcecode/knowledge_base'
know = DocumentService(docs_path='/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/AFAC_LLM/rawdata/观点库.jsonl',
                       vector_store_path=vs_path)
#初始化向量库
#know.init_source_vector()
#加载向量库
know.load_vector_store()
vectore_store = know.vector_store


prompt = """
已知信息：
{knowledge}

你是一个资深金融领域基金研究员，你现在有一条基金结构化信息，请针对这个基金结构化信息，提供一条基金推荐话术。
要求：1.多于10个字同时少于70个字。2.以下列格式输出：基金推荐话术：...。
示例1：
基金结构化信息：“行业：基础化工；利好事件：2017年化肥出口关税税率普遍下调”
基金推荐话术：化肥出口关税下调利好行业盈利，关注尿素、磷肥和复合肥龙头企业，推荐配置基础化工行业基金。

示例2：
基金结构化信息：“行业：传媒；波动：低”
基金推荐话术：本基金关联传媒行业，低风险稳定收益，捕捉成长机会。

基金结构化信息：“{info}”
"""
queries = []
with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/test.json', 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        queries.append(json.loads(line)['text'])

with open('/data/wusiyao/chatglm/chatglm/sourcecode/AFAC/out/prompt_chatglm2/prompt_knowledge.json', 'w+', encoding='utf-8') as f:

    for i in trange(600):
        zhi_q = queries[i*2]
        lihao_q = queries[i*2 + 1]
        result_info, result_event = [], []
        related_docs_zhi = vectore_store.similarity_search_with_score(zhi_q, k=1)[0][0].page_content  # 查找相关信息
        related_docs_lihao = vectore_store.similarity_search_with_score(lihao_q, k=1)[0][0].page_content  # 查找相关信息

        for _ in range(5):
            response, _ = model.chat(tokenizer, prompt.replace("{info}", zhi_q).replace("{knowledge}", related_docs_zhi), history=[])
            result_info.append(response)

        for _ in range(5):
            response, _ = model.chat(tokenizer, prompt.replace("{info}", lihao_q).replace("{knowledge}", related_docs_lihao), history=[])
            result_event.append(response)

        row = {'id':str(i), "info_result":result_info, "event_result":result_event}
        f.write(json.dumps(row, ensure_ascii=False))


